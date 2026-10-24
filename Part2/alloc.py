"""Sensitivity-driven mixed-precision allocation.

1. Profiling (cheap, a few short calibration sequences):
   * for every target Linear and every *reference* bit width b (one spec per bit width), quantize
     that single layer with RTN (in the RHT basis if enabled), run the model from that block to
     the logits and measure KL(p_fp || p_quant) on the calibration tokens  ->  KL_l(b);
   * for every candidate spec c, a Hessian-diagonal proxy of the output error
     e_l(c) = sum_j diag(H)_j * ||W_:,j - Q_c(W)_:,j||^2   (computed in the rotated basis).
2. Cost model: cost_l(c) = KL_l(bits(c)) * e_l(c) / e_l(ref(bits(c)))
   (measured KL for the reference spec of each bit width, scaled by the proxy for other group
   sizes of the same bit width).
3. Allocation: minimise sum_l cost_l(c_l) s.t. sum_l storage_l(c_l) <= budget, solved with a
   Lagrangian relaxation (binary search on lambda) followed by a greedy fill of the slack.
"""
import hashlib
import json
import math
import os

import torch
import torch.nn.functional as F

from common import (TARGET_TYPES, block_linears, capture_block_inputs, free_memory, get_layers, log, run_block,
                    InputStats)
from quant import IncoherentLinear, parse_spec, rtn_quantize


def _cache_path(cache_dir, key):
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"sens_{h}.pt")


@torch.no_grad()
def _final_logprobs(model, h):
    logits = model.lm_head(model.model.norm(h)).float()
    return F.log_softmax(logits, -1)


@torch.no_grad()
def profile_sensitivity(model, calib_ids, specs, ref_specs, cfg, cache_key=None):
    """Return {'kl': {name: {ref_spec: KL}}, 'err': {name: {spec: e}}, 'shape': {name: (out, in)}}.

    ``specs``: all candidate spec strings; ``ref_specs``: one spec string per bit width.
    """
    cache_dir = cfg.get("cache_dir", "cache")
    path = None
    if cache_key is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, dict(cache_key, specs=sorted(specs), refs=sorted(ref_specs)))
        if os.path.exists(path):
            log(f"sensitivity: loading cached profile {path}")
            return torch.load(path, weights_only=False)

    layers = get_layers(model)
    dev = next(model.parameters()).device
    rht = cfg.get("rht", True)
    seed = cfg.get("seed", 0)
    mse_grid = cfg.get("mse_grid", 0)
    inps, kw = capture_block_inputs(model, calib_ids)

    # fp pass: block inputs + Hessian diagonals in the (rotated) basis
    hs = [inps]
    err, shapes, incs = {}, {}, {}
    for i, layer in enumerate(layers):
        lin = block_linears(layer)
        stats = {}
        for t, m in lin.items():
            inc = IncoherentLinear(m.out_features, m.in_features, i, t, seed, dev, enabled=rht)
            incs[(i, t)] = inc
            stats[t] = InputStats(m, inc.input_transform(), diag_only=True)
        out = run_block(layer, hs[-1], kw)
        for t, s in stats.items():
            s.remove()
            name = f"layers.{i}.{t}"
            m = lin[t]
            shapes[name] = tuple(m.weight.shape)
            Wr = incs[(i, t)].to_rotated(m.weight.data.float())
            d = s.H
            err[name] = {c: float(((Wr - rtn_quantize(Wr, parse_spec(c), mse_grid)) ** 2 * d).sum()) for c in specs}
        hs.append(out)
        del stats

    n_seq = inps.shape[0]
    def kl_from(start, h):
        for j in range(start, len(layers)):
            h = run_block(layers[j], h, kw)
        total = 0.0
        for s in range(n_seq):
            lp_q = _final_logprobs(model, h[s:s + 1])
            lp_f = _final_logprobs(model, hs[-1][s:s + 1])
            total += F.kl_div(lp_q, lp_f, log_target=True, reduction="sum").item() / lp_q.shape[1]
            del lp_q, lp_f
        return total / n_seq

    kl = {n: {} for n in shapes}
    for i, layer in enumerate(layers):
        lin = block_linears(layer)
        for t in TARGET_TYPES:
            name = f"layers.{i}.{t}"
            m = lin[t]
            W0 = m.weight.data.clone()
            inc = incs[(i, t)]
            Wr = inc.to_rotated(W0.float())
            for r in ref_specs:
                m.weight.data = inc.from_rotated(rtn_quantize(Wr, parse_spec(r), mse_grid)).to(W0.dtype)
                kl[name][r] = kl_from(i, hs[i])
            m.weight.data = W0
        log(f"  sensitivity block {i:2d}: " + " ".join(
            f"{t[:-5]}={max(kl[f'layers.{i}.{t}'].values()):.3g}" for t in TARGET_TYPES))
    del hs, inps
    free_memory()
    prof = {"kl": kl, "err": err, "shape": shapes}
    if path is not None:
        torch.save(prof, path)
    return prof


def build_costs(prof, specs, ref_specs, extra_bits=None):
    """-> {name: [(spec_str, storage_bits, cost), ...]}"""
    ref_of_bits = {parse_spec(r).bits: r for r in ref_specs}
    table = {}
    for name, (o, i) in prof["shape"].items():
        opts = []
        for c in specs:
            sp = parse_spec(c)
            r = ref_of_bits[sp.bits]
            e_ref = max(prof["err"][name][r], 1e-30)
            cost = prof["kl"][name][r] * prof["err"][name][c] / e_ref
            bits = sp.storage_bits(o, i) + (extra_bits.get(name, 0) if extra_bits else 0)
            opts.append((c, bits, cost))
        table[name] = opts
    return table


def allocate(table, budget_bits, max_iter=100):
    """Minimise total cost subject to total storage <= budget (Lagrangian + greedy fill)."""
    names = list(table)

    def pick(lam):
        choice = {n: min(table[n], key=lambda o: o[2] + lam * o[1]) for n in names}
        return choice, sum(o[1] for o in choice.values())

    hi = 1.0
    choice, used = pick(hi)
    while used > budget_bits and hi < 1e30:
        hi *= 10
        choice, used = pick(hi)
    if used > budget_bits:
        raise ValueError("budget is smaller than the cheapest configuration")
    lo = 1e-40
    for _ in range(max_iter):  # smallest lambda (geometric bisection) that still fits the budget
        mid = math.sqrt(lo * hi)
        c, u = pick(mid)
        if u <= budget_bits:
            hi, choice, used = mid, c, u
        else:
            lo = mid
        if hi / lo < 1 + 1e-6:
            break
    # greedy fill of the remaining budget: best cost reduction per extra bit
    improved = True
    while improved:
        improved = False
        best = None
        for n in names:
            cur = choice[n]
            for o in table[n]:
                db, dc = o[1] - cur[1], cur[2] - o[2]
                if dc > 0 and used + db <= budget_bits:
                    ratio = dc / max(db, 1e-9)
                    if best is None or ratio > best[0]:
                        best = (ratio, n, o, db)
        if best is not None:
            _, n, o, db = best
            choice[n] = o
            used += db
            improved = True
    return {n: parse_spec(o[0]) for n, o in choice.items()}, used


def summarize(assign):
    from collections import Counter
    cnt = Counter(str(s) for s in assign.values())
    by_type = {}
    for n, s in assign.items():
        by_type.setdefault(n.split(".")[-1], Counter())[str(s)] += 1
    log("allocation: " + ", ".join(f"{k}: {v}" for k, v in sorted(cnt.items())))
    for t, c in by_type.items():
        log(f"  {t:10s} " + ", ".join(f"{k}x{v}" for k, v in sorted(c.items())))
