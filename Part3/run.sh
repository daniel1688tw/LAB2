#!/usr/bin/env bash
# Part 3 entry point:  bash run.sh
# Installs requirements.txt, quantizes Qwen/Qwen3-4B with the calibration-free per-tile MX format
# selection (default configuration below) and prints the WikiText-2 PPL measured by the provided
# evaluate.py. Full log: logs/adaptquant.log. Extra arguments are passed to run_ptq.py.
set -uo pipefail
cd "$(dirname "$0")"
if [ "${LAB2_SKIP_INSTALL:-0}" != "1" ]; then
    python -m pip install -q -r requirements.txt > logs_pip_install.txt 2>&1 || { echo "pip install failed (logs_pip_install.txt)" >&2; exit 1; }
fi
mkdir -p logs
out=$(python run_ptq.py --model "${LAB2_MODEL:-Qwen/Qwen3-4B}" "$@" --json logs/results.jsonl 2> logs/adaptquant.log)
echo "$out" | awk '/^avg_bits:/ {printf "Average bits per weight: %.3f\n", $2} /^ppl:/ {printf "PPL: %.2f\n", $2}'
