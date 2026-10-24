"""Calibration-free per-tile MX quantization of Qwen3-4B + WikiText-2 PPL (provided evaluate.py).

python run_ptq.py                       # default AdaptQuant-style configuration
python run_ptq.py --uniform mxfp4       # baseline: every tile MXFP4
python run_ptq.py --budget 5.0 --act none --tile 1x32
"""
import argparse
import contextlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptquant import add_activation_quant, adaptquant  # noqa: E402
from evaluate import DEFAULT_MODEL, eval_ppl_wikitext2, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--formats", default="mxfp4,mxfp6,mxfp8", help="candidate tile formats")
    ap.add_argument("--budget", type=float, default=4.5, help="average bits per weight incl. scales")
    ap.add_argument("--tile", default="128x128", help="rows x cols of a format tile (cols = K, multiple of 32)")
    ap.add_argument("--colw", default="analytic", choices=["analytic", "norm", "uniform"],
                    help="data-free input statistics used to weight the error")
    ap.add_argument("--metric", default="rel", choices=["rel", "abs"])
    ap.add_argument("--no-scale-search", action="store_true", help="plain OCP floor rule for the E8M0 scale")
    ap.add_argument("--act", default="mxfp8", help="activation format (mxfp8 / mxfp6 / mxfp4) or 'none'")
    ap.add_argument("--uniform", default=None, help="baseline: one format for all tiles (e.g. mxfp4, nvfp4)")
    ap.add_argument("--fp16", action="store_true", help="evaluate the unquantized model")
    ap.add_argument("--json", default=None, help="append the result as a json line to this file")
    args = ap.parse_args()

    tr, tc = (int(v) for v in args.tile.lower().split("x"))
    with contextlib.redirect_stdout(sys.stderr):
        torch.manual_seed(0)
        model, tok = load_model(args.model, args.dtype, args.device)
        stats = {"avg_bits": 16.0}
        t0 = time.time()
        if not args.fp16:
            stats = adaptquant(model, formats=tuple(args.formats.split(",")), budget=args.budget, tile=(tr, tc),
                               colw_mode=args.colw, metric=args.metric, scale_search=not args.no_scale_search,
                               uniform=args.uniform, log=lambda *a: print(*a, file=sys.stderr, flush=True))
            if args.act != "none":
                add_activation_quant(model, args.act)
        print(f"quantization time {time.time() - t0:.1f}s", file=sys.stderr)
        ppl = eval_ppl_wikitext2(model, tok)

    print(f"avg_bits: {stats['avg_bits']:.4f}")
    print(f"ppl: {ppl:.4f}")
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(dict(vars(args), ppl=ppl, **stats)) + "\n")


if __name__ == "__main__":
    main()
