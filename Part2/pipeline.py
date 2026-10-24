"""Model-level drivers: sequential GPTQ quantization with optional incoherence processing."""
import torch

from common import (SEQUENTIAL_GROUPS, block_linears, capture_block_inputs, collect_stats, free_memory,
                    get_layers, log, run_block, set_linear_bits)
from quant import IncoherentLinear, gptq_quantize, rtn_quantize

# Linear types that read the same input: the Hessian is collected once per input.
SHARED_INPUT = {"q_proj": "q_proj", "k_proj": "q_proj", "v_proj": "q_proj", "o_proj": "o_proj",
                "gate_proj": "gate_proj", "up_proj": "gate_proj", "down_proj": "down_proj"}


@torch.no_grad()
def quantize_linear_inplace(model, name, layer_idx, module, H, spec, cfg):
    """Quantize one nn.Linear (GPTQ or RTN, optionally in the RHT basis) and record its storage."""
    out_f, in_f = module.weight.shape
    inc = IncoherentLinear(out_f, in_f, layer_idx, name.split(".")[-1], cfg.get("seed", 0),
                           module.weight.device, enabled=cfg.get("rht", True))
    Wr = inc.to_rotated(module.weight.data.float())
    if cfg.get("method", "gptq") == "gptq":
        Hr = inc.rotate_hessian(H)
        Q, loss = gptq_quantize(Wr, Hr, spec, blocksize=cfg.get("blocksize", 128),
                                percdamp=cfg.get("percdamp", 0.01), actorder=cfg.get("actorder", True),
                                mse_grid=cfg.get("mse_grid", 0))
    else:
        Q, loss = rtn_quantize(Wr, spec, cfg.get("mse_grid", 0)), float("nan")
    module.weight.data = inc.from_rotated(Q).to(module.weight.dtype)
    set_linear_bits(model, name, spec.storage_bits(out_f, in_f) + inc.storage_bits(),
                    note=f"{spec}{'+rht' if inc.enabled else ''}")
    return loss


@torch.no_grad()
def gptq_model(model, calib_ids, assign, cfg):
    """Sequentially quantize all target layers.

    ``assign``: {'layers.<i>.<type>': QSpec or None}; layers mapped to None are left untouched.
    """
    layers = get_layers(model)
    inps, kw = capture_block_inputs(model, calib_ids)
    for i, layer in enumerate(layers):
        lin = block_linears(layer)
        for group in SEQUENTIAL_GROUPS:
            todo = [t for t in group if assign.get(f"layers.{i}.{t}") is not None]
            if not todo:
                continue
            sources = sorted({SHARED_INPUT[t] for t in todo})
            stats = collect_stats(layer, sources, inps, kw, bs=cfg.get("fwd_bs", 1))
            for t in todo:
                name = f"layers.{i}.{t}"
                loss = quantize_linear_inplace(model, name, i, lin[t], stats[SHARED_INPUT[t]].H, assign[name], cfg)
                log(f"  {name:22s} {str(assign[name]):10s} gptq-loss {loss:.4g}")
            del stats
            free_memory()
        inps = run_block(layer, inps, kw, bs=cfg.get("fwd_bs", 1))
    del inps
    free_memory()
