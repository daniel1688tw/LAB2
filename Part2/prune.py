"""Structured pruning of MLP intermediate channels with OBS weight compensation.

Removing intermediate channel j of a SwiGLU MLP deletes row j of gate_proj and up_proj and
column j of down_proj; the matrices really shrink, so no mask or index has to be stored.
Channels are chosen greedily with the Optimal-Brain-Surgeon criterion on the input Hessian of
down_proj (as in SlimGPT, Ling et al. 2024):
    score_j = ||W_:,j||^2 / [H^{-1}]_jj
and after every removal the remaining columns of down_proj are updated in closed form
    W <- W - W_:,S [H^{-1}]_SS^{-1} [H^{-1}]_S,:
so that down_proj(x) best reconstructs the original MLP output on the calibration data. This is
a one-shot least-squares reconstruction (no gradient training / fine-tuning).
"""
import torch
import torch.nn as nn

from common import (block_linears, capture_block_inputs, collect_stats, free_memory, get_layers, log, run_block)


@torch.no_grad()
def obs_prune_columns(W, H, n_remove, percdamp=0.01, batch=32):
    """Return (updated W with removed columns zeroed, boolean mask of kept columns)."""
    W = W.clone().float()
    H = H.clone().float()
    n = H.shape[0]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(n, device=H.device)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    alive = torch.ones(n, dtype=torch.bool, device=W.device)
    removed = 0
    while removed < n_remove:
        k = min(batch, n_remove - removed)
        d = torch.diag(Hinv).clamp(min=1e-12)
        score = (W ** 2).sum(0) / d
        score[~alive] = float("inf")
        idx = torch.topk(score, k, largest=False).indices
        Hs_inv = torch.linalg.inv(Hinv[idx][:, idx])
        W -= W[:, idx] @ Hs_inv @ Hinv[idx, :]
        Hinv -= Hinv[:, idx] @ Hs_inv @ Hinv[idx, :]
        W[:, idx] = 0
        Hinv[idx, :] = 0
        Hinv[:, idx] = 0
        alive[idx] = False
        removed += k
    return W, alive


def _shrink_linear(lin, keep, dim):
    """Keep rows (dim=0) or columns (dim=1) of an nn.Linear in place."""
    w = lin.weight.data.index_select(dim, keep).contiguous()
    lin.weight = nn.Parameter(w, requires_grad=False)
    if dim == 0:
        lin.out_features = w.shape[0]
        if lin.bias is not None:
            lin.bias = nn.Parameter(lin.bias.data[keep].contiguous(), requires_grad=False)
    else:
        lin.in_features = w.shape[1]


def layer_schedule(n_layers, n_channels, target_removed_channels, skip_first=2, skip_last=1, align=128,
                   max_ratio=0.6):
    """Number of channels to remove per layer: uniform over the middle layers, none in the first
    ``skip_first`` / last ``skip_last`` layers (the most sensitive ones), multiples of ``align``."""
    active = [i for i in range(n_layers) if skip_first <= i < n_layers - skip_last]
    per = target_removed_channels / len(active)
    per = min(per, max_ratio * n_channels)
    plan = [0] * n_layers
    carry = 0.0
    for i in active:  # distribute the rounding error so the total matches the target
        want = per + carry
        r = int(round(want / align)) * align
        carry = want - r
        plan[i] = r
    return plan


@torch.no_grad()
def prune_mlp_model(model, calib_ids, cfg):
    """Prune intermediate channels of every MLP so that ``cfg['prune_ratio']`` of all target
    Linear parameters are removed. Returns the realised pruning rate."""
    layers = get_layers(model)
    lin0 = block_linears(layers[0])
    hidden, inter = lin0["down_proj"].out_features, lin0["down_proj"].in_features
    total_params = sum(sum(m.weight.numel() for m in block_linears(l).values()) for l in layers)
    per_channel = 3 * hidden  # one row of gate and up + one column of down
    target_channels = cfg["prune_ratio"] * total_params / per_channel
    plan = layer_schedule(len(layers), inter, target_channels, cfg.get("skip_first", 2), cfg.get("skip_last", 1),
                          cfg.get("align", 128), cfg.get("max_layer_ratio", 0.6))
    log(f"pruning plan (channels removed per layer): {plan}")

    inps, kw = capture_block_inputs(model, calib_ids)
    removed_params = 0
    for i, layer in enumerate(layers):
        if plan[i] > 0:
            lin = block_linears(layer)
            stats = collect_stats(layer, ["down_proj"], inps, kw)
            W, alive = obs_prune_columns(lin["down_proj"].weight.data, stats["down_proj"].H, plan[i],
                                         cfg.get("percdamp", 0.01), cfg.get("obs_batch", 32))
            del stats
            keep = torch.nonzero(alive).flatten()
            lin["down_proj"].weight.data = W.to(lin["down_proj"].weight.dtype)
            _shrink_linear(lin["gate_proj"], keep, 0)
            _shrink_linear(lin["up_proj"], keep, 0)
            _shrink_linear(lin["down_proj"], keep, 1)
            removed_params += plan[i] * per_channel
            log(f"  layer {i:2d}: kept {keep.numel()}/{inter} MLP channels")
            free_memory()
        inps = run_block(layer, inps, kw)
    del inps
    free_memory()
    return removed_params / total_params
