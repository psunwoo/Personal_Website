import numpy as np
import torch

from rcf import synthetic
from rcf.compare import holdout
from rcf.models import KINDS, SetModel
from rcf.stats import paired_bootstrap, pass_rule

torch.set_num_threads(1)


def _batch(n=6, L=5, d=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, L, d, generator=g)
    m = torch.ones(n, L, dtype=torch.bool)
    m[0, 3:] = False
    m[1, 1:] = False
    return x * m.unsqueeze(-1), m


def test_permutation_and_padding_invariance():
    x, m = _batch()
    perm = torch.tensor([2, 0, 1, 4, 3])
    for kind in KINDS:
        torch.manual_seed(0)
        f = SetModel(7, 3, kind).eval()
        for p in f.parameters():  # move off the "starts additive" initialisation
            torch.nn.init.normal_(p, std=0.3)
        z = f(x, m)
        xp, mp = x[2:, perm], m[2:, perm]
        assert torch.allclose(z[2:], f(xp, mp), atol=1e-4), kind
        junk = x.clone()
        junk[0, 3:] = 99.0  # padded slots must not matter
        assert torch.allclose(z, f(junk, m), atol=1e-4), kind


def test_second_order_models_start_at_the_additive_model():
    x, m = _batch()
    for kind in ("poly2", "poly3", "pair", "gate", "scope_l", "scope_r", "gate_poly2", "sgate"):
        torch.manual_seed(1)
        f = SetModel(7, 3, kind).eval()
        z = f(x, m)
        A = (f.phi(x) * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
        assert torch.allclose(z, f.lin(A), atol=1e-5), kind


def test_additive_model_is_additive():
    """With equal set sizes the order-1 logits are an average of single-object contributions."""
    torch.manual_seed(2)
    f = SetModel(7, 3, "add").eval()
    x = torch.randn(4, 3, 7)
    m = torch.ones(4, 3, dtype=torch.bool)
    A = f.lin(f.phi(x)).mean(1)
    assert torch.allclose(f(x, m), A, atol=1e-5)


def test_planted_pair_term_has_no_additive_component():
    _, info = synthetic.make(20000, 0.3, seed=5)
    data, _ = synthetic.make(20000, 0.3, seed=5)
    # an additive predictor sees only per-object functions; a rich proxy: mean of object features and their squares
    m = data.mask[..., None]
    n = data.mask.sum(1, keepdims=True)
    F = np.concatenate([(data.X * m).sum(1) / n, ((data.X ** 2) * m).sum(1) / n, np.ones((len(data), 1))], 1)
    two = info["raw_two"]
    beta, *_ = np.linalg.lstsq(F, two, rcond=None)
    r2 = 1 - ((two - F @ beta) ** 2).sum() / ((two - two.mean(0)) ** 2).sum()
    assert r2 < 0.02, r2


def test_bootstrap_and_pass_rule():
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.5, size=(3, 4000))
    same = paired_bootstrap(base, base + rng.normal(0, 0.01, size=base.shape), B=500)
    assert same.lo < 0 < same.hi
    better = paired_bootstrap(base, base - 0.05 + rng.normal(0, 0.01, size=base.shape), B=500)
    assert better.lo > 0 and abs(better.mean - 0.05) < 0.005
    assert pass_rule(better, L1=1.0, Lceil=0.8)["passed"]  # 0.05 of 0.2 headroom = 25 %
    assert not pass_rule(better, L1=1.0, Lceil=0.0)["passed"]  # 5 % of headroom: significant but too small
    grouped = paired_bootstrap(base, base - 0.05, B=200, groups=np.arange(4000) // 10)
    assert grouped.lo > 0


def test_planted_interaction_is_recovered_and_null_gives_null():
    out = {}
    for eta in (0.0, 0.5):
        tr, _ = synthetic.make(3000, eta, seed=11, task="reg")
        va, _ = synthetic.make(600, eta, seed=12, task="reg")
        te, _ = synthetic.make(2000, eta, seed=13, task="reg")
        r = holdout(tr, va, te, ["add", "pair"], seeds=1, tune=False, verbose=False, max_epochs=25)
        out[eta] = paired_bootstrap(r["losses"]["add"].mean(-1), r["losses"]["pair"].mean(-1), B=300)
    assert out[0.5].lo > 0.02, out[0.5]
    assert out[0.0].mean < 0.01, out[0.0]


def test_cca_recovers_shared_directions():
    from rcf.rg import CCA

    rng = np.random.default_rng(0)
    s = rng.normal(size=(6000, 3))  # 3 shared latent directions
    X = np.concatenate([s + 0.3 * rng.normal(size=s.shape), rng.normal(size=(6000, 7))], 1) @ rng.normal(size=(10, 10))
    Y = np.concatenate([s + 0.3 * rng.normal(size=s.shape), rng.normal(size=(6000, 5))], 1) @ rng.normal(size=(8, 8))
    c = CCA(X[:4000], Y[:4000])
    rho = c.heldout_rho(X[4000:], Y[4000:])
    assert (rho[:3] > 0.85).all() and (np.abs(rho[3:]) < 0.1).all(), rho


def test_scoped_operator_acts_on_one_side_only():
    """A damping factor on the first object changes the others' contributions under scope_l, not under scope_r."""
    torch.manual_seed(3)
    x = torch.randn(1, 3, 7)
    x[0, :, -2] = torch.tensor([0.0, 0.5, 1.0])  # positions
    m = torch.ones(1, 3, dtype=torch.bool)
    for kind, changes in (("scope_l", True), ("scope_r", False)):
        f = SetModel(7, 2, kind).eval()
        base = f(x, m)
        orig = f.factors
        f.factors = lambda xx, mm, o=orig: o(xx, mm) * torch.tensor([0.0, 1.0, 1.0]).view(1, 3, 1)  # object 0 becomes a full damper
        assert (not torch.allclose(f(x, m), base, atol=1e-6)) == changes, kind
