"""Stream EmoSet-118K shards from the Hugging Face Hub and cache features.

For every image we store
  - metadata: label, brightness, colorfulness, scene, facial_expression,
    human_action, object list (EmoSet's own attributes);
  - DINOv2 ViT-S/14 CLS embeddings of the image at three block scales, each
    block embedded *in isolation* (no information from other blocks):
        4x4 tiles (56 px), 3x3 tiles (70 px), 2x2 tiles (112 px), 1x1 (whole, 224 px);
  - a CLIP ViT-B/32 embedding of the whole image (language-aligned reference).

Shards are deleted after processing, so peak disk use is one shard (~1 GB).
The job is resumable: finished shards are skipped.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoModel, CLIPModel

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_image(b: bytes, size: int = 224) -> torch.Tensor:
    im = Image.open(io.BytesIO(b)).convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)


def tiles(x: torch.Tensor, g: int) -> torch.Tensor:
    """(B,3,H,W) -> (B*g*g,3,H/g,W/g), row-major tile order."""
    B, C, H, W = x.shape
    t = x.unfold(2, H // g, H // g).unfold(3, W // g, W // g)  # B,C,g,g,h,w
    return t.permute(0, 2, 3, 1, 4, 5).reshape(B * g * g, C, H // g, W // g)


@torch.inference_mode()
def embed_batch(imgs: torch.Tensor, dino, clip) -> dict[str, np.ndarray]:
    x = imgs.float() / 255.0  # B,3,224,224
    xd = (x - MEAN) / STD
    B = x.shape[0]

    def cls(t: torch.Tensor, g: int) -> np.ndarray:
        h = dino(pixel_values=t).last_hidden_state[:, 0]
        return h.reshape(B, g * g, -1).numpy().astype(np.float16)

    out = {}
    out["d1"] = cls(xd, 1)[:, 0]  # whole image, 224 px
    out["d2"] = cls(tiles(xd, 2), 2)  # 4 tiles of 112 px
    out["d3"] = cls(tiles(F.interpolate(xd, size=210, mode="bilinear", antialias=True), 3), 3)  # 9 tiles of 70 px
    out["d4"] = cls(tiles(xd, 4), 4)  # 16 tiles of 56 px
    xc = (x - CLIP_MEAN) / CLIP_STD
    out["clip"] = clip.visual_projection(clip.vision_model(pixel_values=xc).pooler_output).numpy().astype(np.float16)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Woleek/EmoSet-118K")
    ap.add_argument("--out", default="data/features/emoset")
    ap.add_argument("--tmp", default="data/raw/emoset_tmp")
    ap.add_argument("--splits", default="test,val,train")
    ap.add_argument("--max-shards", type=int, default=10**6, help="per split")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="debug: rows per shard")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet"))
    dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

    def shard_key(f: str) -> int:
        m = re.findall(r"(\d+)", Path(f).stem)
        return int(m[0]) if m else 0

    for split in args.splits.split(","):
        sf = sorted([f for f in files if f"/{split}-" in f], key=shard_key)[: args.max_shards]
        for f in sf:
            name = Path(f).stem
            dst = out / f"{name}.npz"
            if dst.exists():
                continue
            t0 = time.time()
            local = hf_hub_download(args.repo, f, repo_type="dataset", local_dir=args.tmp)
            t_dl = time.time() - t0
            pf = pq.ParquetFile(local)
            meta = {k: [] for k in ["image_id", "label", "brightness", "colorfulness", "facial_expression", "human_action", "scene", "object"]}
            feats = {k: [] for k in ["d1", "d2", "d3", "d4", "clip"]}
            n = 0
            for rb in pf.iter_batches(batch_size=args.batch):
                rows = rb.to_pylist()
                imgs, keep = [], []
                for r in rows:
                    try:
                        imgs.append(load_image(r["image"]["bytes"]))
                        keep.append(r)
                    except Exception as e:  # corrupt image: skip, but say so
                        print(f"  skip {r.get('image_id')}: {e}", flush=True)
                if not imgs:
                    continue
                e = embed_batch(torch.stack(imgs), dino, clip)
                for k in feats:
                    feats[k].append(e[k])
                for r in keep:
                    for k in meta:
                        v = r.get(k)
                        meta[k].append(json.dumps(v) if k == "object" else v)
                n += len(keep)
                if args.limit and n >= args.limit:
                    break
            np.savez(
                dst,
                **{k: np.concatenate(v) for k, v in feats.items()},
                **{k: np.array(["" if x is None else x for x in v]) if k not in ("label", "brightness", "colorfulness") else np.array(v, dtype=np.float32) for k, v in meta.items()},
            )
            os.remove(local)
            dt = time.time() - t0
            print(f"{name}: {n} imgs, download {t_dl:.0f}s, total {dt:.0f}s, {n / max(dt - t_dl, 1e-9):.1f} img/s", flush=True)


if __name__ == "__main__":
    main()
