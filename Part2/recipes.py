"""Mixed-precision GPTQ recipe shared by cases 1, 2 and 4."""
from alloc import allocate, build_costs, profile_sensitivity, summarize
from common import BASE_BITS, compressed_linear_bits, get_calib_data, log, target_linears
from pipeline import gptq_model
from quant import parse_spec

# Candidate specs shared by the quantization cases. The sensitivity profile is always computed
# for the union of these (and the case's own lists), so cases 1 and 2 share one cached profile.
ALL_SPECS = ["w4g128a", "w4g256a", "w4cha", "w3g64a", "w3g128a", "w3g256a", "w3cha", "w2g64a", "w2g128a"]
ALL_REFS = ["w4g128a", "w3g128a", "w2g128a"]


def filter_specs(specs, wbits):
    """Keep candidate specs whose bit width is within +-1 of ``wbits`` (all of them if wbits is None)."""
    if wbits is None:
        return list(specs)
    kept = [s for s in specs if abs(parse_spec(s).bits - wbits) <= 1]
    if not kept:
        raise ValueError(f"no candidate spec within +-1 bit of wbits={wbits}: {specs}")
    return kept


def rht_overhead(model, cfg):
    if not cfg.get("rht", True):
        return {}
    return {n: (m.in_features + 64) + (m.out_features + 64) for n, m in target_linears(model).items()}


def mixed_precision_gptq(model, tokenizer, rate, cfg, state_tag="fp"):
    """Allocate a quantization spec per target layer so that the compression rate is >= ``rate``,
    then quantize the model with sequential GPTQ."""
    specs = filter_specs(cfg["specs"], cfg.get("wbits"))
    refs = [r for r in cfg["ref_specs"] if parse_spec(r).bits in {parse_spec(s).bits for s in specs}]
    names = list(target_linears(model))
    extra = rht_overhead(model, cfg)

    # bits of everything that is not re-quantized here (nothing for cases 1/2) are already fixed
    budget = (1 - rate) * BASE_BITS * model._base_linear_numel
    if len(specs) == 1:
        assign = {n: parse_spec(specs[0]) for n in names}
    else:
        prof_ids = get_calib_data(tokenizer, cfg["sens_nsamples"], cfg["sens_seqlen"], cfg["seed"] + 1, cfg["dataset"])
        key = dict(model=getattr(model.config, "_name_or_path", ""), state=state_tag, rht=cfg.get("rht", True),
                   seed=cfg["seed"], n=cfg["sens_nsamples"], L=cfg["sens_seqlen"], data=cfg["dataset"],
                   mse_grid=cfg.get("mse_grid", 0))
        prof_specs = sorted(set(ALL_SPECS) | set(cfg["specs"]))
        prof_refs = sorted(set(ALL_REFS) | set(cfg["ref_specs"]))
        log(f"profiling layer sensitivity: specs={prof_specs} refs={prof_refs}")
        prof = profile_sensitivity(model, prof_ids, prof_specs, prof_refs, cfg, cache_key=key)
        log(f"allocation candidates: {specs} (reference specs {refs})")
        table = build_costs(prof, specs, refs, extra)
        assign, used = allocate(table, budget)
        log(f"allocated {used / 8 / 2**20:.1f} MiB of a {budget / 8 / 2**20:.1f} MiB budget")
    summarize(assign)

    calib = get_calib_data(tokenizer, cfg["nsamples"], cfg["seqlen"], cfg["seed"], cfg["dataset"])
    log(f"GPTQ on {calib.shape[0]} x {calib.shape[1]} calibration tokens (rht={cfg.get('rht', True)})")
    gptq_model(model, calib, assign, cfg)
    rate_now = 1 - compressed_linear_bits(model) / (BASE_BITS * model._base_linear_numel)
    if rate_now < rate:
        log(f"WARNING: compression rate {rate_now:.4%} is below the target {rate:.4%}")
    return assign
