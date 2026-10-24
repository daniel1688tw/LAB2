"""Simulated (fake-quantized) microscaling formats.

OCP Microscaling (MX) v1.0: a block of 32 consecutive elements along the reduction (K) dimension
shares one E8M0 scale X = 2^e (8 bits); every element is a narrow float:
    MXFP8 = E4M3 (max 448)   MXFP6 = E2M3 (max 7.5) or E3M2 (max 28)   MXFP4 = E2M1 (max 6)
NVFP4 (NVIDIA Blackwell): blocks of 16 E2M1 elements with an FP8-E4M3 block scale and one FP32
per-tensor scale.

All functions take / return float32 tensors whose values lie exactly on the format's grid.
"""
import math

import torch

# name -> (exponent bits, mantissa bits, max normal value)
ELEMENTS = {
    "e2m1": (2, 1, 6.0),
    "e2m3": (2, 3, 7.5),
    "e3m2": (3, 2, 28.0),
    "e4m3": (4, 3, 448.0),   # OCP E4M3 / torch.float8_e4m3fn
    "e5m2": (5, 2, 57344.0),
}

# format name -> description
FORMATS = {
    "mxfp4": dict(kind="mx", elem="e2m1", block=32, elem_bits=4, scale_bits=8),
    "mxfp6": dict(kind="mx", elem="e2m3", block=32, elem_bits=6, scale_bits=8),
    "mxfp6e3m2": dict(kind="mx", elem="e3m2", block=32, elem_bits=6, scale_bits=8),
    "mxfp8": dict(kind="mx", elem="e4m3", block=32, elem_bits=8, scale_bits=8),
    "nvfp4": dict(kind="nv", elem="e2m1", block=16, elem_bits=4, scale_bits=8),
}


def bits_per_element(fmt):
    f = FORMATS[fmt]
    return f["elem_bits"] + f["scale_bits"] / f["block"]


def emax_elem(elem):
    """Exponent of the largest power of two <= max normal (OCP 'emax' of the element type)."""
    return int(math.floor(math.log2(ELEMENTS[elem][2])))


def round_minifloat(x, elem):
    """Round to the nearest value of the element format (ties to even), saturating at +-max."""
    ebits, mbits, max_val = ELEMENTS[elem]
    bias = 2 ** (ebits - 1) - 1
    emin = 1 - bias                            # exponent of the smallest normal number
    a = x.abs().clamp(max=max_val)
    e = torch.floor(torch.log2(a.clamp(min=1e-38))).clamp(min=emin)
    step = torch.exp2(e - mbits)               # spacing of the grid in this binade (subnormals: emin)
    q = torch.round(a / step) * step
    return torch.sign(x) * q.clamp(max=max_val)


def _blocks(x, block):
    """[..., K] -> [..., K/block, block] (pads K with zeros if needed). Returns view and pad."""
    k = x.shape[-1]
    pad = (-k) % block
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], x.shape[-1] // block, block), pad


def _unblock(xb, pad):
    x = xb.reshape(*xb.shape[:-2], -1)
    return x[..., :x.shape[-1] - pad] if pad else x


def mx_quantize(x, fmt="mxfp4", scale_search=False, weight=None):
    """Fake-quantize ``x`` along its last dimension with an MX format.

    scale_search=False: OCP rule  e = floor(log2(amax)) - emax_elem.
    scale_search=True : per block pick e or e+1 (the latter avoids clipping the block maximum),
                        whichever gives the smaller (optionally ``weight``-weighted) squared error.
    ``weight``: per-element importance broadcastable to x (e.g. per-column weights), only used by
    the scale search. Purely a function of x -> calibration-free.
    """
    f = FORMATS[fmt]
    assert f["kind"] == "mx", fmt
    xb, pad = _blocks(x.float(), f["block"])
    amax = xb.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126))) - emax_elem(f["elem"])
    e = e.clamp(-127, 127)
    q = round_minifloat(xb / torch.exp2(e), f["elem"]) * torch.exp2(e)
    if scale_search:
        e1 = (e + 1).clamp(max=127)
        q1 = round_minifloat(xb / torch.exp2(e1), f["elem"]) * torch.exp2(e1)
        if weight is not None:
            wb, _ = _blocks(weight.float().expand_as(x), f["block"])
            err0, err1 = ((q - xb) ** 2 * wb).sum(-1, keepdim=True), ((q1 - xb) ** 2 * wb).sum(-1, keepdim=True)
        else:
            err0, err1 = ((q - xb) ** 2).sum(-1, keepdim=True), ((q1 - xb) ** 2).sum(-1, keepdim=True)
        q = torch.where(err1 < err0, q1, q)
    return _unblock(q, pad)


def nvfp4_quantize(x, scale_search=False, weight=None):
    """Fake NVFP4: per-tensor FP32 scale g, per-16 block E4M3 scale, E2M1 elements."""
    f = FORMATS["nvfp4"]
    x = x.float()
    g = (x.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30)
    xb, pad = _blocks(x, f["block"])
    amax = xb.abs().amax(-1, keepdim=True)
    s = round_minifloat(amax / 6.0 / g, "e4m3").clamp(min=2.0 ** -9) * g
    q = round_minifloat(xb / s, "e2m1") * s
    return _unblock(q, pad)


def fake_quantize(x, fmt, scale_search=False, weight=None):
    if FORMATS[fmt]["kind"] == "nv":
        return nvfp4_quantize(x, scale_search, weight)
    return mx_quantize(x, fmt, scale_search, weight)
