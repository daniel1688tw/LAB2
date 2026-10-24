"""Case 2: compression rate >= 81%, PPL < 20.

Strategy: same pipeline as case 1 (RHT + GPTQ + sensitivity-driven mixed precision) with a
~3-bit budget: candidate specs span 2/3/4 bits and group sizes from 64 to a whole row, so the
allocator can trade scale overhead and bit width between sensitive and robust layers.
"""
from common import get_config, log, report, set_seed
from recipes import mixed_precision_gptq

DEFAULTS = dict(
    nsamples=128, seqlen=2048, dataset="wikitext2",
    sens_nsamples=8, sens_seqlen=512,
    rht=True, method="gptq", actorder=True, mse_grid=20, percdamp=0.01, blocksize=128,
    specs=["w4g128a", "w4cha", "w3g64a", "w3g128a", "w3g256a", "w3cha", "w2g64a", "w2g128a"],
    ref_specs=["w4g128a", "w3g128a", "w2g128a"],
    wbits=None, seed=0, cache_dir="cache",
)


def compress(model, tokenizer, rate, wbits=None, seed=0):
    cfg = get_config("case2", DEFAULTS)
    cfg["seed"] = seed
    if wbits is not None:
        cfg["wbits"] = wbits
    set_seed(seed)
    log(f"case 2 config: {cfg}")
    mixed_precision_gptq(model, tokenizer, rate, cfg)
    report(model, "case 2")
    return model
