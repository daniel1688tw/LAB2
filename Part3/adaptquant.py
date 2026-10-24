"""Calibration-free per-tile MX format selection (AdaptQuant-style reproduction).

For every target Linear W [out, in] (K = in is the MX block dimension):
1. fake-quantize W with every candidate format (MXFP4 / MXFP6 / MXFP8, optionally NVFP4), with an
   optional per-block E8M0 exponent search (e vs e+1);
2. split W into hardware-aligned tiles (default 128 x 128: a multiple of the 32-element MX block
   along K and of the 128-row MMA tile along N) and compute, per tile and format, the
   input-weighted squared error  sum_ij c_j (W_ij - Q_ij)^2, normalised by the layer's
   "output energy" sum_ij c_j W_ij^2;
   c_j = E[x_j^2] is *derived from the weights only* (no data):
     q/k/v_proj, gate/up_proj : x = RMSNorm(h)  ->  c_j = gamma_j^2
     o_proj                    : x = attention-weighted values -> c = diag(W_v diag(gamma^2) W_v^T)
                                 (repeated over the GQA query heads)
     down_proj                 : x = silu(g) * u with g, u ~ N(0, diag(W diag(gamma^2) W^T))
                                 -> c = E[silu(g)^2] E[u^2] (Gauss-Hermite quadrature)
3. choose one format per tile minimising the total error under an average bit budget
   (bits include the 8-bit block scales and the per-tile format id), with a global Lagrangian
   relaxation over all tiles of the model;
4. write the composed fake-quantized weight back (FP16), optionally quantize every Linear input
   (activation) to MXFP8 on the fly.
Nothing depends on calibration data, so the procedure is fully target-model calibration-free.
"""
import math

import numpy as np
import torch

from mxformats import FORMATS, bits_per_element, fake_quantize, mx_quantize

TARGET_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def block_linears(layer):
    a, m = layer.self_attn, layer.mlp
    return {"q_proj": a.q_proj, "k_proj": a.k_proj, "v_proj": a.v_proj, "o_proj": a.o_proj,
            "gate_proj": m.gate_proj, "up_proj": m.up_proj, "down_proj": m.down_proj}


# --------------------------------------------------------------------------------------
# data-free input statistics
# --------------------------------------------------------------------------------------
_GH = np.polynomial.hermite_e.hermegauss(64)  # nodes / weights for E over N(0, 1)


def _gauss_expect(fn, std):
    """E[fn(std * z)], z ~ N(0,1), elementwise over the tensor ``std``."""
    z = torch.tensor(_GH[0], dtype=torch.float32, device=std.device)
    w = torch.tensor(_GH[1] / _GH[1].sum(), dtype=torch.float32, device=std.device)
    return (fn(std.unsqueeze(-1) * z) * w).sum(-1)


@torch.no_grad()
def input_second_moments(layer, config, mode="analytic"):
    """{type: c [in_features]} with c_j ~ E[x_j^2] of each Linear's input, from the weights only."""
    lin = block_linears(layer)
    dev = lin["q_proj"].weight.device
    out = {}
    if mode == "uniform":
        return {t: torch.ones(m.in_features, device=dev) for t, m in lin.items()}
    g1 = layer.input_layernorm.weight.float() ** 2
    g2 = layer.post_attention_layernorm.weight.float() ** 2
    for t in ("q_proj", "k_proj", "v_proj"):
        out[t] = g1
    for t in ("gate_proj", "up_proj"):
        out[t] = g2
    if mode == "norm":
        out["o_proj"] = torch.ones(lin["o_proj"].in_features, device=dev)
        out["down_proj"] = torch.ones(lin["down_proj"].in_features, device=dev)
        return out
    # o_proj input: values of the kv head used by every query head
    wv = lin["v_proj"].weight.float()
    var_v = (wv ** 2 * g1).sum(1)                              # [n_kv * head_dim]
    hd = config.head_dim if getattr(config, "head_dim", None) else config.hidden_size // config.num_attention_heads
    rep = config.num_attention_heads // config.num_key_value_heads
    var_v = var_v.view(config.num_key_value_heads, 1, hd).expand(-1, rep, -1).reshape(-1)
    out["o_proj"] = var_v
    # down_proj input: silu(gate x) * (up x)
    var_g = (lin["gate_proj"].weight.float() ** 2 * g2).sum(1)
    var_u = (lin["up_proj"].weight.float() ** 2 * g2).sum(1)
    e_silu2 = _gauss_expect(lambda v: (v * torch.sigmoid(v)) ** 2, var_g.sqrt())
    out["down_proj"] = e_silu2 * var_u
    return {t: v / v.mean().clamp(min=1e-30) for t, v in out.items()}


# --------------------------------------------------------------------------------------
# per-tile errors
# --------------------------------------------------------------------------------------
def tile_sums(x, tr, tc):
    """[out, in] -> [out/tr, in/tc] sums (tr | out, tc | in)."""
    o, i = x.shape
    return x.reshape(o // tr, tr, i // tc, tc).sum((1, 3))


@torch.no_grad()
def quantize_all_formats(W, formats, colw, scale_search):
    w = colw.unsqueeze(0) if colw is not None else None
    return {f: fake_quantize(W, f, scale_search, w) for f in formats}


@torch.no_grad()
def layer_tile_errors(W, formats, colw, tile, scale_search, metric="rel"):
    tr, tc = tile
    W = W.float()
    c = colw.unsqueeze(0)
    errs = []
    for f, Q in quantize_all_formats(W, formats, colw, scale_search).items():
        errs.append(tile_sums((W - Q) ** 2 * c, tr, tc).flatten())
    errs = torch.stack(errs, 1)  # [tiles, formats]
    if metric == "rel":
        errs = errs / (W ** 2 * c).sum().clamp(min=1e-30)
    return errs


# --------------------------------------------------------------------------------------
# global allocation
# --------------------------------------------------------------------------------------
def allocate(errs, tile_elems, bpe, budget_bits_per_elem, id_bits):
    """errs [T, F], tile_elems [T], bpe [F] -> format index per tile (Lagrangian, global)."""
    total = tile_elems.sum()
    bits = tile_elems.unsqueeze(1) * bpe.unsqueeze(0)               # [T, F]
    budget = budget_bits_per_elem * total - id_bits * errs.shape[0]

    def pick(lam):
        idx = torch.argmin(errs + lam * bits, 1)
        return idx, bits.gather(1, idx.unsqueeze(1)).sum()

    cheapest = bits.min(1).values.sum()
    if cheapest > budget:
        raise ValueError(f"budget {budget_bits_per_elem} bits/elem is below the cheapest format")
    lo, hi = 1e-30, 1.0
    idx, used = pick(hi)
    while used > budget:
        hi *= 10
        idx, used = pick(hi)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        i2, u2 = pick(mid)
        if u2 <= budget:
            hi, idx, used = mid, i2, u2
        else:
            lo = mid
        if hi / lo < 1 + 1e-7:
            break
    return idx, (used + id_bits * errs.shape[0]) / total


# --------------------------------------------------------------------------------------
# model-level driver
# --------------------------------------------------------------------------------------
@torch.no_grad()
def adaptquant(model, formats=("mxfp4", "mxfp6", "mxfp8"), budget=4.5, tile=(128, 128), colw_mode="analytic",
               metric="rel", scale_search=True, uniform=None, log=print):
    """Quantize all target Linear weights in place. Returns a stats dict."""
    layers = model.model.layers
    if uniform is not None:
        formats = (uniform,)
    for f in formats:
        assert tile[1] % FORMATS[f]["block"] == 0, "tile width must be a multiple of the block size"
    bpe = torch.tensor([bits_per_element(f) for f in formats], dtype=torch.float64)
    id_bits = math.ceil(math.log2(len(formats))) if len(formats) > 1 else 0

    # pass 1: tile errors of every layer / format
    entries, all_errs, all_elems = [], [], []
    colws = {}
    for li, layer in enumerate(layers):
        cw = input_second_moments(layer, model.config, colw_mode)
        for t, m in block_linears(layer).items():
            W = m.weight.data
            o, i = W.shape
            tr, tc = min(tile[0], o), min(tile[1], i)
            assert o % tr == 0 and i % tc == 0, (t, W.shape, tile)
            colws[(li, t)] = cw[t]
            if len(formats) > 1:
                e = layer_tile_errors(W, formats, cw[t], (tr, tc), scale_search, metric).double().cpu()
            else:
                e = torch.zeros((o // tr) * (i // tc), 1, dtype=torch.float64)
            entries.append((li, t, (tr, tc), e.shape[0]))
            all_errs.append(e)
            all_elems.append(torch.full((e.shape[0],), tr * tc, dtype=torch.float64))
    errs, elems = torch.cat(all_errs), torch.cat(all_elems)
    if len(formats) > 1:
        choice, avg_bits = allocate(errs, elems, bpe, budget, id_bits)
    else:
        choice = torch.zeros(errs.shape[0], dtype=torch.long)
        avg_bits = float(bpe[0])
    log(f"average bits per weight (incl. scales and tile ids): {float(avg_bits):.4f}")

    # pass 2: write back the composed fake-quantized weights
    stats = {"avg_bits": float(avg_bits), "format_share": {}, "type_bits": {}}
    share = torch.zeros(len(formats), dtype=torch.float64)
    type_bits = {t: [0.0, 0.0] for t in TARGET_TYPES}
    pos = 0
    for (li, t, (tr, tc), n_tiles) in entries:
        m = block_linears(layers[li])[t]
        W = m.weight.data.float()
        o, i = W.shape
        idx = choice[pos:pos + n_tiles].to(W.device).view(o // tr, i // tc)
        pos += n_tiles
        mask_full = idx.repeat_interleave(tr, 0).repeat_interleave(tc, 1)
        Wq = torch.zeros_like(W)
        for fi, f in enumerate(formats):
            sel = mask_full == fi
            if sel.any():
                Q = fake_quantize(W, f, scale_search, colws[(li, t)].unsqueeze(0))
                Wq = torch.where(sel, Q, Wq)
            n = int((idx == fi).sum())
            share[fi] += n * tr * tc
            type_bits[t][0] += n * tr * tc * float(bpe[fi])
        type_bits[t][1] += o * i
        m.weight.data = Wq.to(m.weight.dtype)
    stats["format_share"] = {f: float(share[k] / share.sum()) for k, f in enumerate(formats)}
    stats["type_bits"] = {t: v[0] / v[1] for t, v in type_bits.items() if v[1]}
    log("format share: " + ", ".join(f"{f} {s:.2%}" for f, s in stats["format_share"].items()))
    log("bits per layer type: " + ", ".join(f"{t} {b:.2f}" for t, b in stats["type_bits"].items()))
    per_layer = {}
    pos = 0
    for (li, t, _, n_tiles) in entries:
        per_layer.setdefault(li, []).append(bpe[choice[pos:pos + n_tiles]].mean().item())
        pos += n_tiles
    log("bits per decoder layer: " + " ".join(f"{li}:{np.mean(v):.2f}" for li, v in per_layer.items()))
    return stats


def add_activation_quant(model, fmt="mxfp8"):
    """Fake-quantize the input of every target Linear (dynamic, per 32-element block along K)."""
    handles = []

    def hook(mod, inp):
        x = inp[0]
        return (mx_quantize(x, fmt).to(x.dtype),) + tuple(inp[1:])

    for layer in model.model.layers:
        for m in block_linears(layer).values():
            handles.append(m.register_forward_pre_hook(hook))
    return handles
