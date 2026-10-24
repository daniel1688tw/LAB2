"""Format / allocation unit tests (CPU).  python tests/test_mx.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adaptquant import allocate, tile_sums  # noqa: E402
from mxformats import bits_per_element, fake_quantize, mx_quantize, round_minifloat  # noqa: E402

torch.manual_seed(0)

# E4M3 rounding == torch.float8_e4m3fn (round-to-nearest-even, saturate)
x = torch.randn(200000) * torch.exp2(torch.randint(-12, 9, (200000,)).float())
x = x.clamp(-448, 448)
ref = x.to(torch.float8_e4m3fn).float()
got = round_minifloat(x, "e4m3")
assert torch.equal(got, ref), (x[got != ref][:5], got[got != ref][:5], ref[got != ref][:5])
# E5M2 as well
ref = x.to(torch.float8_e5m2).float()
assert torch.equal(round_minifloat(x, "e5m2"), ref)

# E2M1 / E2M3 / E3M2 grids
grid = lambda e: sorted(set(round_minifloat(torch.linspace(0, 40, 400001), e).tolist()))  # noqa: E731
assert grid("e2m1") == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
g = grid("e2m3")
assert len(g) == 32 and g[1] == 0.125 and g[-1] == 7.5
g = grid("e3m2")
assert len(g) == 32 and g[1] == 0.0625 and g[-1] == 28

# MX block structure: every 32-block = power-of-two scale * element grid
W = torch.randn(64, 256) * 0.02
W[3, 7] = 0.5
for fmt, elem_max in (("mxfp4", 6), ("mxfp6", 7.5), ("mxfp8", 448)):
    Q = mx_quantize(W, fmt).view(64, 8, 32)
    amax = W.view(64, 8, 32).abs().amax(-1)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - {"mxfp4": 2, "mxfp6": 2, "mxfp8": 8}[fmt])
    assert (Q.abs() / scale.unsqueeze(-1) <= elem_max + 1e-6).all()
errs = {f: ((fake_quantize(W, f) - W) ** 2).mean().item() for f in ("mxfp4", "nvfp4", "mxfp6", "mxfp8")}
assert errs["mxfp8"] < errs["mxfp6"] < errs["nvfp4"] < errs["mxfp4"], errs
# scale search never hurts
e0 = ((mx_quantize(W, "mxfp4") - W) ** 2).sum()
e1 = ((mx_quantize(W, "mxfp4", scale_search=True) - W) ** 2).sum()
assert e1 <= e0
print("formats ok:", {k: f"{v:.3g}" for k, v in errs.items()}, f"scale search {e0:.4g} -> {e1:.4g}")
assert bits_per_element("mxfp4") == 4.25 and bits_per_element("nvfp4") == 4.5 and bits_per_element("mxfp8") == 8.25

# allocation respects the budget and prefers upgrading high-error tiles
errs = torch.rand(1000, 3).double().sort(1, descending=True).values
errs[:100, 0] *= 100
bpe = torch.tensor([4.25, 6.25, 8.25], dtype=torch.float64)
elems = torch.full((1000,), 16384.0, dtype=torch.float64)
idx, avg = allocate(errs, elems, bpe, 4.75, 2)
assert avg <= 4.75 + 1e-9 and avg > 4.6, avg
assert (idx[:100] > 0).float().mean() > 0.9
assert torch.equal(tile_sums(torch.ones(256, 64), 128, 32), torch.full((2, 2), 4096.0))
print("allocation ok, avg bits", float(avg))
print("ALL MX TESTS PASSED")

# block rotation keeps the full-precision product unchanged
from adaptquant import block_rotate, hadamard  # noqa: E402
H = hadamard(32, "cpu")
W = torch.randn(64, 256)
x = torch.randn(10, 256)
assert torch.allclose(block_rotate(W, H) @ block_rotate(x, H).t(), W @ x.t(), atol=1e-4)
print("block rotation ok")
