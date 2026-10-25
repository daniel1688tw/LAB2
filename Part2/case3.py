"""Case 3: SVD compression rate >= 20%, overall compression rate = 50%, PPL < 50.

Strategy:
1. truncation-aware whitened SVD (SVD-LLM) of every target layer, with the same rank ratio on
   every layer so that the low-rank factors hold (1 - svd_ratio) of the original parameters
   (SVD compression rate = svd_ratio, measured at 16 bit); a global energy-based allocator is
   also implemented (`rank_alloc="global"`) but is worse in practice (see ablation.sh);
2. the factors (and the few layers the allocator keeps dense) are quantized to ``wbits``
   (default 8, per-row symmetric, ~lossless); since 8-bit everywhere would overshoot the 50%
   target (0.8 * 8/16 = 40% of the size -> 60%), the first layers are kept in float16 until the
   overall rate lands just above ``rate``.
"""
from common import BASE_BITS, get_calib_data, get_config, log, record_base, report, set_seed, target_linears
from svd import allocate_ranks, collect_spectra, factor_bits, svd_compress_model, uniform_ranks

DEFAULTS = dict(
    nsamples=128, seqlen=2048, dataset="wikitext2",
    svd_ratio=0.20,         # SVD compression rate (parameter reduction from the low-rank factors)
    svd_margin=0.002,       # aim slightly above the required 20%
    rank_alloc="uniform",   # "uniform": same ratio everywhere, "global": energy-based allocation across layers
    min_rank_frac=0.10,     # never truncate a factorized layer below 10% of its full rank
    factor_bits=None,       # defaults to wbits (8)
    type_weights={},        # optional per-type weights of the truncation loss, e.g. {"down_proj": 2.0}
    seed=0,
)


def compress(model, tokenizer, rate, wbits=None, seed=0):
    if not hasattr(model, "_base_linear_numel"):  # in case the caller did not call common.record_base
        record_base(model)
    cfg = get_config("case3", DEFAULTS)
    cfg["seed"] = seed
    set_seed(seed)
    fbits = cfg["factor_bits"] or wbits or 8
    log(f"case 3 config: {cfg} factor_bits={fbits}")
    lin = target_linears(model)
    base_params = model._base_linear_numel

    calib = get_calib_data(tokenizer, cfg["nsamples"], cfg["seqlen"], seed, cfg["dataset"])
    log("pass 1: whitened spectra of the uncompressed model")
    spectra = collect_spectra(model, calib, cfg)
    weights = {n: cfg["type_weights"].get(n.split(".")[-1], 1.0) for n in spectra}
    keep = (1 - cfg["svd_ratio"] - cfg["svd_margin"]) * base_params
    if cfg["rank_alloc"] == "uniform":
        ranks, kept_params = uniform_ranks(spectra, keep / base_params)
    else:
        ranks, kept_params = allocate_ranks(spectra, keep, cfg["min_rank_frac"], weights)
    svd_rate = 1 - kept_params / base_params
    n_svd = sum(r is not None for r in ranks.values())
    log(f"rank allocation: {n_svd}/{len(ranks)} layers factorized, SVD compression rate {svd_rate:.2%}")

    # factor precision: quantize to fbits, then restore float16 (first layers first) while the
    # overall compression rate stays >= rate
    # (layers the allocator keeps dense are quantized with the same per-row format)
    fb = {n: fbits for n in ranks}

    def total_bits(fbm):
        t = 0
        for n, m in lin.items():
            r = ranks[n]
            if r is None:
                t += factor_bits(*m.weight.shape, fbm[n])
            else:
                o, i = m.weight.shape
                t += factor_bits(o, r, fbm[n]) + factor_bits(r, i, fbm[n])
        return t

    budget = (1 - rate) * BASE_BITS * base_params
    if total_bits(fb) > budget:
        raise ValueError("the SVD factors do not fit the budget; lower factor_bits or raise svd_ratio")
    for n in fb:  # model order
        trial = dict(fb, **{n: 16})
        if total_bits(trial) <= budget:
            fb = trial
    log(f"layers kept in float16: {sum(v == 16 for v in fb.values())}, quantized to {fbits} bit: "
        f"{sum(v < 16 for v in fb.values())}")

    log("pass 2: sequential whitened SVD + factor quantization")
    svd_compress_model(model, calib, ranks, fb, cfg)
    model._extra_stats["SVD compression rate (16-bit parameter reduction)"] = f"{svd_rate:.2%}"
    report(model, "case 3")
    return model
