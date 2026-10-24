"""Case 1: compression rate > 75%, PPL < 15.

Strategy: incoherence processing (randomized Hadamard transform) + GPTQ with a per-layer
mixed-precision allocation of {4, 3}-bit specs driven by measured KL sensitivity.
"""
from common import get_config, log, report, set_seed
from recipes import mixed_precision_gptq

DEFAULTS = dict(
    nsamples=128, seqlen=2048, dataset="wikitext2",   # GPTQ calibration (WikiText-2 *train* split)
    sens_nsamples=8, sens_seqlen=512,                  # sensitivity profiling
    rht=True, method="gptq", actorder=True, mse_grid=20, percdamp=0.01, blocksize=128,
    specs=["w4g128a", "w4g256a", "w4cha", "w3g128a", "w3g256a", "w3cha"],
    ref_specs=["w4g128a", "w3g128a"],
    wbits=None, seed=0, cache_dir="cache",
)


def compress(model, tokenizer, rate, wbits=None, seed=0):
    cfg = get_config("case1", DEFAULTS)
    cfg["seed"] = seed
    if wbits is not None:
        cfg["wbits"] = wbits
    set_seed(seed)
    log(f"case 1 config: {cfg}")
    mixed_precision_gptq(model, tokenizer, rate, cfg)
    report(model, "case 1")
    return model
