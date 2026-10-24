#!/usr/bin/env bash
# Experiments for the report (results appended to logs/results.jsonl, one json line per run).
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
M="--model ${LAB2_MODEL:-Qwen/Qwen3-4B} --json logs/results.jsonl"
r() { echo "== $*"; python run_ptq.py $M "$@" 2> "logs/sweep_$(echo "$*" | tr ' /' '__').log" | tr '\n' ' '; echo; }

r --fp16
# uniform-format baselines (weight-only, then with MXFP8 activations)
for f in mxfp4 nvfp4 mxfp6 mxfp8; do r --uniform $f --act none; done
r --uniform mxfp4
# AdaptQuant-style: bit budget sweep
for b in 4.375 4.5 4.75 5.0; do r --budget $b; done
# ablations at 4.5 bits
r --budget 4.5 --act none
r --budget 4.5 --no-scale-search
r --budget 4.5 --colw uniform
r --budget 4.5 --colw norm
r --budget 4.5 --metric abs
r --budget 4.5 --tile 1x32
r --budget 4.5 --tile 128x32
r --budget 4.5 --tile 256x256
r --budget 4.5 --formats mxfp4,nvfp4,mxfp6,mxfp8 --tile 128x128
r --budget 4.5 --formats mxfp4,mxfp6
# data-free block Hadamard rotation inside the MX blocks (online activation rotation)
r --uniform mxfp4 --act none --rotate 32
r --budget 4.5 --rotate 32
r --budget 4.25 --rotate 32 --formats mxfp4,mxfp6,mxfp8
