"""Truncation-aware whitened SVD (SVD-LLM, Wang et al. 2024) with global rank allocation.

For a Linear y = W x with input second moment H = E[x x^T] = S S^T, the output error of a
rank-k approximation of W S is exactly the sum of the dropped squared singular values of W S
(||(W - W_k) X||_F^2 = sum_{i>k} sigma_i^2), so truncating in the whitened basis is optimal for
the layer output. The low-rank layer is stored as two factors
    A = U_k sqrt(Sigma_k)              [out, k]
    B = sqrt(Sigma_k) V_k^T S^{-1}     [k, in]
and the layer becomes W ~= A @ B (evaluated here as one dense fake weight; a deployed model
runs two thin GEMMs).
"""
import math

import torch

from common import (SEQUENTIAL_GROUPS, block_linears, capture_block_inputs, collect_stats, free_memory, get_layers,
                    log, run_block, set_linear_bits)
from pipeline import SHARED_INPUT
from quant import QSpec, rtn_quantize


@torch.no_grad()
def whitening(H, eps=1e-6):
    """H = S S^T via eigendecomposition (robust to rank-deficient H). Returns S, S^{-1}."""
    H = H.double() if H.device.type == "cpu" or H.shape[0] <= 4096 else H.float()
    H = (H + H.t()) / 2
    evals, evecs = torch.linalg.eigh(H)
    evals = evals.clamp(min=eps * evals.max().clamp(min=1e-12))
    sq = evals.sqrt()
    S = evecs * sq
    S_inv = (evecs / sq).t()
    return S.float(), S_inv.float()


@torch.no_grad()
def whitened_svd(W, H):
    S, S_inv = whitening(H)
    U, s, Vh = torch.linalg.svd(W.float() @ S, full_matrices=False)
    return U, s, Vh, S_inv


def dense_rank_limit(m, n):
    """Largest rank for which the two factors are smaller than the dense matrix."""
    return (m * n) // (m + n)


def allocate_ranks(spectra, keep_params, min_rank_frac=0.05, weights=None):
    """Global rank allocation.

    spectra: {name: (sigma [r], m, n)}. Minimises sum_l w_l * (dropped energy fraction of layer l)
    subject to total parameters <= keep_params; every layer is either kept dense (no SVD) or
    stored as factors with rank k <= dense_rank_limit.
    Returns {name: rank or None (dense)}.
    """
    info = {}
    for name, (s, m, n) in spectra.items():
        e = (s.double() ** 2).cpu()
        frac_kept = torch.cumsum(e, 0) / e.sum()                          # energy kept with rank k+1
        loss = torch.cat([torch.ones(1, dtype=torch.float64), 1 - frac_kept])  # loss(k), k = 0..r
        w = 1.0 if weights is None else weights.get(name, 1.0)
        kmax = min(dense_rank_limit(m, n), len(s))
        kmin = max(1, int(min_rank_frac * min(m, n)))
        ks = torch.arange(kmin, kmax + 1)
        info[name] = (ks, w * loss[ks].clamp(min=0), ks.double() * (m + n), m * n)

    def pick(lam):
        choice, total = {}, 0
        for name, (ks, loss, params, dense) in info.items():
            obj = loss + lam * params
            j = int(torch.argmin(obj))
            if lam * dense <= obj[j]:  # dense: zero loss
                choice[name], p = None, dense
            else:
                choice[name], p = int(ks[j]), float(params[j])
            total += p
        return choice, total

    lo, hi = 1e-30, 1.0
    choice, total = pick(hi)
    while total > keep_params:
        hi *= 10
        choice, total = pick(hi)
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        c, t = pick(mid)
        if t <= keep_params:
            hi, choice, total = mid, c, t
        else:
            lo = mid
        if hi / lo < 1 + 1e-9:
            break
    return choice, total


@torch.no_grad()
def collect_spectra(model, calib_ids, cfg):
    """Pass 1 over the uncompressed model: whitened singular values of every target layer."""
    layers = get_layers(model)
    inps, kw = capture_block_inputs(model, calib_ids)
    spectra = {}
    for i, layer in enumerate(layers):
        lin = block_linears(layer)
        stats = collect_stats(layer, sorted(set(SHARED_INPUT.values())), inps, kw)
        for t, m in lin.items():
            _, s, _, _ = whitened_svd(m.weight.data, stats[SHARED_INPUT[t]].H)
            spectra[f"layers.{i}.{t}"] = (s.cpu(), m.out_features, m.in_features)
        del stats
        free_memory()
        inps = run_block(layer, inps, kw)
    return spectra


def factor_bits(rows, cols, bits):
    return 16 * rows * cols if bits >= 16 else QSpec(bits, -1, sym=True).storage_bits(rows, cols)


@torch.no_grad()
def svd_compress_model(model, calib_ids, ranks, factor_bits_map, cfg):
    """Pass 2 (sequential): replace every layer with rank != None by its quantized low-rank factors.

    The whitening uses the activations of the already-compressed preceding layers.
    """
    layers = get_layers(model)
    inps, kw = capture_block_inputs(model, calib_ids)
    for i, layer in enumerate(layers):
        lin = block_linears(layer)
        for group in SEQUENTIAL_GROUPS:
            todo = [t for t in group if ranks.get(f"layers.{i}.{t}") is not None]
            if not todo:
                continue
            stats = collect_stats(layer, sorted({SHARED_INPUT[t] for t in todo}), inps, kw)
            for t in todo:
                name = f"layers.{i}.{t}"
                m = lin[t]
                k = ranks[name]
                U, s, Vh, S_inv = whitened_svd(m.weight.data, stats[SHARED_INPUT[t]].H)
                sq = s[:k].sqrt()
                A = U[:, :k] * sq
                B = (sq.unsqueeze(1) * Vh[:k]) @ S_inv
                fb = factor_bits_map.get(name, 16)
                if fb < 16:
                    A = rtn_quantize(A, QSpec(fb, -1, sym=True))
                    B = rtn_quantize(B, QSpec(fb, -1, sym=True))
                m.weight.data = (A @ B).to(m.weight.dtype)
                bits = factor_bits(A.shape[0], k, fb) + factor_bits(k, B.shape[1], fb)
                set_linear_bits(model, name, bits, note=f"svd r{k} {fb}b")
                energy = float((s[:k] ** 2).sum() / (s ** 2).sum())
                log(f"  {name:22s} rank {k:5d}/{min(m.weight.shape):5d} factors {fb:2d}b  kept energy {energy:.4f}")
            del stats
            free_memory()
        inps = run_block(layer, inps, kw)
    del inps
    free_memory()
