#!/usr/bin/env bash
# Main entry point of Part 2:  bash run.sh
#   1. installs requirements.txt
#   2. compresses Qwen/Qwen3-4B for each of the four cases (always from the original model)
#   3. evaluates each case with the provided run_eval.py
#   4. prints only the Compression Ratio and PPL of every case (full logs: logs/caseN.log)
#
# Optional environment variables: LAB2_MODEL (default Qwen/Qwen3-4B), LAB2_CASES (default "1 2 3 4"),
# LAB2_SKIP_INSTALL=1, LAB2_RUN_EVAL (default run_eval.py), LAB2_CASE<N>_CFG='{"key": value}' (see caseN.py)
set -uo pipefail
cd "$(dirname "$0")"

if [ "${LAB2_SKIP_INSTALL:-0}" != "1" ]; then
    if ! python -m pip install -q -r requirements.txt > logs_pip_install.txt 2>&1; then
        echo "pip install failed, see logs_pip_install.txt" >&2
        exit 1
    fi
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="${LAB2_MODEL:-Qwen/Qwen3-4B}"
RUN_EVAL="${LAB2_RUN_EVAL:-run_eval.py}"
mkdir -p logs

# case  target-rate  wbits
#   1   > 75%        4      (aim 75.5%)
#   2   >= 81%       3      (aim 81.1%)
#   3   = 50%        8      (SVD rate >= 20%, factors 8-bit, overall just above 50%)
#   4   >= 85%       3      (pruning rate >= 20%, aim 85.1%)
declare -A RATE=([1]=0.755 [2]=0.811 [3]=0.5 [4]=0.851)
declare -A WBITS=([1]=4 [2]=3 [3]=8 [4]=3)

first=1
for c in ${LAB2_CASES:-1 2 3 4}; do
    out=$(python "$RUN_EVAL" --model "$MODEL" --case "$c" --rate "${RATE[$c]}" --wbits "${WBITS[$c]}" 2> "logs/case$c.log")
    ppl=$(echo "$out" | awk '/^ppl:/ {printf "%.2f", $2}')
    rate=$(echo "$out" | awk '/^comp_rate:/ {v=$2; sub("%", "", v); printf "%.1f%%", v}')
    [ $first -eq 1 ] || echo
    first=0
    echo "Case $c"
    if [ -z "$ppl" ]; then
        echo "FAILED (see logs/case$c.log)"
        continue
    fi
    echo "Compression Ratio: $rate"
    echo "PPL: $ppl"
done
