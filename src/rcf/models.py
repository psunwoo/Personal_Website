"""Permutation-invariant set models with a controlled body order.

All models share the same skeleton

    e_i = phi(x_i)                      one-body embedding, K-dim
    A   = mean_i e_i                    pooled one-body basis (intensive)
    z   = gain(n) * [ W A + higher(A or pairs) ] + bias(n)

where n is the set size.  gain(n), bias(n) let every model, including the
additive one, choose its own normalisation (mean, sum, anything between), so a
pure size effect is never counted as an interaction.

kind      body order   higher-order term
--------  -----------  ------------------------------------------------------
add       1            none
poly2     <= 2         Q2 [(U A) * (V A)]            (ACE-style product basis)
poly3     <= 3         poly2 + Q3 [(U A)*(V A)*(W A)]
pair      <= 2         mean_{i != j} psi(x_i, x_j)   (explicit, Janossy k=2)
deepsets  any          rho(A), rho an MLP
settrans  any          self-attention over objects
"""
from __future__ import annotations

import torch
import torch.nn as nn

KINDS = ("add", "poly2", "poly3", "pair", "deepsets", "settrans")
BODY_ORDER = {"add": 1, "poly2": 2, "pair": 2, "poly3": 3, "deepsets": None, "settrans": None}


def mlp(d_in: int, hidden: int, d_out: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, d_out))


class SetModel(nn.Module):
    def __init__(self, d_in: int, n_out: int, kind: str = "add", K: int = 64, hidden: int = 128,
                 rank: int = 16, dropout: float = 0.0):
        super().__init__()
        assert kind in KINDS, kind
        self.kind, self.n_out = kind, n_out
        self.phi = mlp(d_in, hidden, K, dropout)
        self.lin = nn.Linear(K, n_out)
        self.size = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 2 * n_out))
        nn.init.zeros_(self.size[-1].weight)
        nn.init.zeros_(self.size[-1].bias)
        if kind in ("poly2", "poly3"):
            self.U2, self.V2 = nn.Linear(K, rank), nn.Linear(K, rank)
            self.Q2 = nn.Linear(rank, n_out, bias=False)
            nn.init.zeros_(self.Q2.weight)  # start exactly at the additive model
        if kind == "poly3":
            self.U3, self.V3, self.W3 = nn.Linear(K, rank), nn.Linear(K, rank), nn.Linear(K, rank)
            self.Q3 = nn.Linear(rank, n_out, bias=False)
            nn.init.zeros_(self.Q3.weight)
        if kind == "pair":
            kp = max(K // 2, 8)
            self.phi_p = mlp(d_in, hidden, kp, dropout)
            self.psi = mlp(2 * kp, hidden, n_out, dropout)
            nn.init.zeros_(self.psi[-1].weight)
            nn.init.zeros_(self.psi[-1].bias)
        if kind == "deepsets":
            self.rho = mlp(K, hidden, n_out, dropout)
        if kind == "settrans":
            self.inp = nn.Linear(d_in, K)
            layer = nn.TransformerEncoderLayer(K, 4, 2 * K, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        n = m.sum(1, keepdim=True)
        nc = n.clamp(min=1.0)
        if self.kind == "settrans":
            pad = ~mask
            pad = pad & ~(pad.all(1, keepdim=True))  # an empty set would give NaN attention
            h = self.enc(self.inp(x), src_key_padding_mask=pad)
            A = (h * m.unsqueeze(-1)).sum(1) / nc
            z = self.lin(A)
        else:
            A = (self.phi(x) * m.unsqueeze(-1)).sum(1) / nc
            z = self.lin(A)
            if self.kind in ("poly2", "poly3"):
                z = z + self.Q2(self.U2(A) * self.V2(A))
            if self.kind == "poly3":
                z = z + self.Q3(self.U3(A) * self.V3(A) * self.W3(A))
            if self.kind == "deepsets":
                z = self.rho(A)
            if self.kind == "pair":
                z = z + self.pair_term(x, mask)
        s = self.size(torch.cat([torch.log(nc), 1.0 / nc], dim=1))
        gain, bias = s[:, : self.n_out], s[:, self.n_out:]
        return torch.exp(gain) * z + bias

    def pair_term(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = self.phi_p(x)  # B,L,kp
        B, L, k = e.shape
        ei = e.unsqueeze(2).expand(B, L, L, k)
        ej = e.unsqueeze(1).expand(B, L, L, k)
        psi = self.psi(torch.cat([ei, ej], -1))  # ordered pairs; summed over both orders below
        m = mask.float()
        pm = m.unsqueeze(2) * m.unsqueeze(1) * (1.0 - torch.eye(L, device=x.device)).unsqueeze(0)
        denom = pm.sum((1, 2)).clamp(min=1.0).unsqueeze(-1)
        return (psi * pm.unsqueeze(-1)).sum((1, 2)) / denom
