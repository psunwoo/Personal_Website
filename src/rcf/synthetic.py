"""Synthetic object sets with a planted, Hoeffding-pure pair interaction.

logits z = mean_i f(x_i) + lam * mean_{i<j} h(x_i) * h(x_j),   E_x[h(x)] = 0

Because h is centred under the object distribution and objects are drawn
i.i.d., the pair term has no additive (one-body) component: an additive model
cannot absorb any of it.  Both terms are scaled to unit logit variance, so
eta = lam^2 / (1 + lam^2) is the share of logit variance carried by the
interaction.
"""
from __future__ import annotations

import numpy as np

from .data import SetData


def make(n: int, eta: float, seed: int = 0, teacher_seed: int = 0, d: int = 16, C: int = 9, n_cat: int = 20,
         size: tuple[int, int] = (2, 10), n_ann: int = 6, temp: float = 1.0, task: str = "dist") -> tuple[SetData, dict]:
    tr = np.random.default_rng(teacher_seed)  # the world: fixed across dataset draws
    proto = tr.normal(size=(n_cat, d))
    B1, W1 = tr.normal(size=(d, 32)) / np.sqrt(d), tr.normal(size=(32, C))
    B2, W2 = tr.normal(size=(d, 12)) / np.sqrt(d), tr.normal(size=(12, C))

    def objects(rng, m):
        return (proto[rng.integers(n_cat, size=m)] + 0.3 * rng.normal(size=(m, d))).astype(np.float32)

    ref = objects(np.random.default_rng(10_000 + teacher_seed), 50_000)
    h_mean = np.tanh(ref @ B2).mean(0)

    rng = np.random.default_rng(seed)
    L = size[1]
    N = rng.integers(size[0], size[1] + 1, size=n)
    X = objects(rng, n * L).reshape(n, L, d)
    M = np.arange(L)[None, :] < N[:, None]
    X = X * M[..., None]
    m = M[..., None].astype(np.float64)
    f = np.tanh(X @ B1) @ W1  # n,L,C
    one = (f * m).sum(1) / N[:, None]
    h = (np.tanh(X @ B2) - h_mean) * m  # n,L,12 centred
    s1, s2 = h.sum(1), (h ** 2).sum(1)
    npairs = (N * (N - 1) / 2)[:, None]
    two = (0.5 * (s1 ** 2 - s2) / npairs) @ W2  # mean over i<j of h_i*h_j, then mixed into C logits

    # scale both parts to unit variance using a fixed reference draw (not this sample)
    if not hasattr(make, "_scale") or make._scale[0] != (teacher_seed, d, C, n_cat, size):
        make._scale = ((teacher_seed, d, C, n_cat, size), None)
        _, info = make(20_000, 0.0, seed=987_654, teacher_seed=teacher_seed, d=d, C=C, n_cat=n_cat, size=size,
                       n_ann=n_ann, temp=temp, task=task)
        make._scale = ((teacher_seed, d, C, n_cat, size), (info["raw_one"].std(0), info["raw_two"].std(0)))
    scale = make._scale[1]
    lam = np.sqrt(eta / (1.0 - eta)) if eta > 0 else 0.0
    if scale is None:
        z = one
    else:
        z = (one - one.mean(0)) / scale[0] + lam * two / scale[1]
    info = dict(raw_one=one, raw_two=two, lam=lam)
    if task == "dist":
        p = np.exp(temp * z - (temp * z).max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        y = np.stack([rng.multinomial(n_ann, pi) for pi in p]) / n_ann
        info["p_true"] = p
        return SetData(X.astype(np.float32), M, y.astype(np.float32), "dist"), info
    noise = rng.normal(size=z.shape) * 0.5
    info["z_true"], info["noise_sd"] = z, 0.5
    return SetData(X.astype(np.float32), M, (z + noise).astype(np.float32), "reg"), info
