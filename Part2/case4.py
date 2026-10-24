"""Case 4: pruning rate >= 20%, overall compression rate >= 85%, PPL < 50.

Strategy:
1. structured pruning of MLP intermediate channels (OBS selection + closed-form compensation of
   down_proj) removing ``prune_ratio`` of all target-layer parameters (matrices really shrink);
2. RHT + GPTQ with sensitivity-driven mixed precision on the pruned model so that the overall
   compression rate reaches ``rate``.
"""
from common import get_calib_data, get_config, log, report, set_seed
from prune import prune_mlp_model
from recipes import mixed_precision_gptq

DEFAULTS = dict(
    nsamples=128, seqlen=2048, dataset="wikitext2",
    sens_nsamples=8, sens_seqlen=512,
    prune_ratio=0.21, skip_first=2, skip_last=1, align=128, max_layer_ratio=0.6, obs_batch=32,
    rht=True, method="gptq", actorder=True, mse_grid=20, percdamp=0.01, blocksize=128,
    specs=["w4g128a", "w4cha", "w3g64a", "w3g128a", "w3g256a", "w3cha", "w2g64a", "w2g128a"],
    ref_specs=["w4g128a", "w3g128a", "w2g128a"],
    wbits=None, seed=0, cache_dir="cache",
)


def compress(model, tokenizer, rate, wbits=None, seed=0):
    cfg = get_config("case4", DEFAULTS)
    cfg["seed"] = seed
    if wbits is not None:
        cfg["wbits"] = wbits
    set_seed(seed)
    log(f"case 4 config: {cfg}")
    calib = get_calib_data(tokenizer, cfg["nsamples"], cfg["seqlen"], seed, cfg["dataset"])
    pruning_rate = prune_mlp_model(model, calib, cfg)
    del calib
    log(f"pruning rate: {pruning_rate:.2%}")
    model._extra_stats["pruning rate (parameters removed)"] = f"{pruning_rate:.2%}"
    tag = f"pruned-{cfg['prune_ratio']}-{cfg['skip_first']}-{cfg['skip_last']}-{cfg['align']}"
    mixed_precision_gptq(model, tokenizer, rate, cfg, state_tag=tag)
    report(model, "case 4")
    return model
