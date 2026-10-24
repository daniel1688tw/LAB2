"""Unit tests of the compression building blocks (CPU, a few seconds).  python tests/test_units.py"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from alloc import allocate  # noqa: E402
from prune import layer_schedule, obs_prune_columns  # noqa: E402
from quant import RHT, IncoherentLinear, QSpec, gptq_quantize, parse_spec, rtn_quantize  # noqa: E402
from svd import allocate_ranks, whitened_svd  # noqa: E402

torch.manual_seed(0)


def correlated_inputs(n, d, outlier=True):
    A = torch.randn(d, d) / math.sqrt(d)
    X = torch.randn(n, d) @ A + 0.1 * torch.randn(n, d)
    if outlier:
        X[:, :3] *= 20  # a few outlier channels, like LLM activations
    return X


def test_rht():
    for n in (256, 768, 2560, 1024):
        r = RHT(n, seed=1, device="cpu")
        x = torch.randn(5, n)
        y = r.forward(x)
        assert torch.allclose(r.inverse(y), x, atol=1e-5)
        assert torch.allclose(y.norm(dim=1), x.norm(dim=1), rtol=1e-5)
    W = torch.randn(64, 768)
    X = correlated_inputs(500, 768)
    H = X.t() @ X
    inc = IncoherentLinear(64, 768, 0, "q_proj", 0, "cpu")
    Wr, Hr = inc.to_rotated(W), inc.rotate_hessian(H)
    assert torch.allclose(inc.from_rotated(Wr), W, atol=1e-4)
    # output error is invariant: tr(W H W^T) == tr(W' H' W'^T)
    assert math.isclose(torch.trace(W @ H @ W.t()).item(), torch.trace(Wr @ Hr @ Wr.t()).item(), rel_tol=1e-4)
    # outliers are spread out: max |W'| relative to RMS is much smaller
    W[3, 5] = 50
    Wr = inc.to_rotated(W)
    assert Wr.abs().max() / Wr.pow(2).mean().sqrt() < 0.3 * W.abs().max() / W.pow(2).mean().sqrt()
    print("rht ok")


def test_specs():
    s = QSpec(4, 128, sym=False)
    assert s.storage_bits(10, 256) == 10 * 256 * 4 + 10 * 2 * (16 + 4)
    s = QSpec(3, -1, sym=True)
    assert s.storage_bits(10, 256) == 10 * 256 * 3 + 10 * 16
    for txt in ("w4g128a", "w3chs", "w2g64a"):
        assert str(parse_spec(txt)) == txt
    W = torch.randn(32, 256)
    for txt in ("w4g128a", "w3cha", "w2g64s"):
        sp = parse_spec(txt)
        Q = rtn_quantize(W, sp)
        levels = max(torch.unique(Q[r, :sp.groups(256)[1]]).numel() for r in range(32))
        assert levels <= 2 ** sp.bits, (txt, levels)
    # clipping search never increases the error
    for txt in ("w3g128a", "w2cha"):
        sp = parse_spec(txt)
        assert ((rtn_quantize(W, sp, 20) - W) ** 2).sum() <= ((rtn_quantize(W, sp) - W) ** 2).sum() * 1.0001
    print("specs ok")


def test_gptq():
    d, o = 256, 64
    X = correlated_inputs(2000, d)
    H = 2 * X.t() @ X / X.shape[0]
    W = torch.randn(o, d)
    for txt in ("w4g128a", "w3cha", "w2g64a"):
        sp = parse_spec(txt)
        Qr = rtn_quantize(W, sp)
        Qg, _ = gptq_quantize(W, H, sp, actorder=True)
        Qg2, _ = gptq_quantize(W, H, sp, actorder=False)
        e = lambda Q: torch.trace((Q - W) @ H @ (Q - W).t()).item()  # noqa: E731
        assert e(Qg) < e(Qr) and e(Qg2) < e(Qr), (txt, e(Qr), e(Qg), e(Qg2))
        # the GPTQ output stays on the quantization grid of its static groups
        ng, gs = sp.groups(d)
        assert max(torch.unique(Qg[r, :gs]).numel() for r in range(o)) <= 2 ** sp.bits
        print(f"gptq {txt}: rtn {e(Qr):.4g} gptq {e(Qg):.4g} (no act-order {e(Qg2):.4g})")


def test_obs_prune():
    d, o = 256, 64
    X = correlated_inputs(3000, d, outlier=False)
    H = 2 * X.t() @ X / X.shape[0]
    W = torch.randn(o, d)
    Wp, alive = obs_prune_columns(W, H, 64)
    assert alive.sum() == d - 64 and (Wp[:, ~alive] == 0).all()
    naive = W.clone()
    naive[:, ~alive] = 0
    e = lambda Q: torch.trace((Q - W) @ H @ (Q - W).t()).item()  # noqa: E731
    assert e(Wp) < e(naive), (e(Wp), e(naive))
    print(f"obs prune: compensated {e(Wp):.4g} vs zeroing {e(naive):.4g}")
    plan = layer_schedule(36, 9728, 0.21 * 36 * 100.9e6 / (3 * 2560), 2, 1, 128)
    assert plan[0] == plan[1] == plan[35] == 0 and all(p % 128 == 0 for p in plan)
    assert abs(sum(plan) * 3 * 2560 / (36 * 100.9e6) - 0.21) < 0.005
    for target in (100.0, 500.0, 1300.0):  # the realised count never falls below a feasible target
        plan = layer_schedule(6, 768, target, 2, 1, 64)
        assert sum(plan) >= target and all(p % 64 == 0 for p in plan), (target, plan)


def test_svd():
    d, o = 128, 96
    X = correlated_inputs(1000, d)
    H = X.t() @ X
    W = torch.randn(o, d)
    U, s, Vh, S_inv = whitened_svd(W, H)
    k = len(s)
    rec = (U[:, :k] * s[:k]) @ Vh[:k] @ S_inv
    assert torch.allclose(rec, W, atol=1e-3)
    # truncation error in the output equals the dropped energy
    k = 40
    Wk = (U[:, :k] * s[:k]) @ Vh[:k] @ S_inv
    out_err = ((Wk - W) @ X.t()).pow(2).sum().item()
    assert math.isclose(out_err, (s[k:] ** 2).sum().item(), rel_tol=1e-2)
    # whitened truncation beats plain truncation on the output error
    U2, s2, Vh2 = torch.linalg.svd(W, full_matrices=False)
    plain = ((U2[:, :k] * s2[:k]) @ Vh2[:k] - W) @ X.t()
    assert out_err < plain.pow(2).sum().item()
    spectra = {"a": (torch.linspace(10, 0.1, 64), 64, 256), "b": (torch.logspace(1, -3, 64), 64, 256)}
    ranks, total = allocate_ranks(spectra, 0.5 * 2 * 64 * 256)
    assert total <= 0.5 * 2 * 64 * 256 and ranks["b"] is not None and ranks["b"] < (ranks["a"] or 64)
    print("svd ok", ranks)


def test_allocate():
    table = {f"l{i}": [("w4g128a", 400, 1.0 * (i + 1)), ("w3g128a", 300, 3.0 * (i + 1)), ("w2g128a", 200, 50.0 * (i + 1))]
             for i in range(4)}
    for budget in (800, 1000, 1300, 1600):
        assign, used = allocate(table, budget)
        assert used <= budget
        # exhaustive check of optimality on this small problem
        import itertools
        best = min((sum(o[2] for o in combo), combo) for combo in itertools.product(*table.values())
                   if sum(o[1] for o in combo) <= budget)
        got = sum(next(o[2] for o in table[n] if o[0] == str(s)) for n, s in assign.items())
        assert got <= best[0] * 1.0001, (budget, got, best)
    print("allocate ok")


if __name__ == "__main__":
    test_rht()
    test_specs()
    test_gptq()
    test_obs_prune()
    test_svd()
    test_allocate()
    print("ALL UNIT TESTS PASSED")
