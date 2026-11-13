"""Encoder registry with an on-disk cache.  Every string is embedded in isolation (one-body)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODELS = {
    "glove": "sentence-transformers/average_word_embeddings_glove.6B.300d",  # static co-occurrence vectors
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # small transformer sentence encoder
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def embed_texts(texts: list[str], model: str, cache_dir: str = "data/cache", batch: int = 256) -> np.ndarray:
    """(len(texts), d) float32.  Cached per (model, exact list of texts)."""
    key = hashlib.sha1(("\x1f".join(texts)).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"emb_{model}_{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(MODELS[model], device="cpu")
    if model == "glove":  # the packaged tokenizer drops stop words; keep them (e.g. "give up", "not bad")
        enc[0].tokenizer.stop_words = set()
    E = enc.encode(texts, batch_size=batch, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, E)
    return E
