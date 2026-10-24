#!/usr/bin/env bash
# Optional ablations for the report (not needed for grading). Each line = one run of run_eval.py.
# usage: bash ablation.sh            results are appended to logs/ablation.txt
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
MODEL="${LAB2_MODEL:-Qwen/Qwen3-4B}"
RUN_EVAL="${LAB2_RUN_EVAL:-run_eval.py}"

run() {  # run <label> <case> <rate> <wbits> <json cfg>
    local label=$1 c=$2 rate=$3 wbits=$4 cfg=$5
    local out
    out=$(env "LAB2_CASE${c}_CFG=$cfg" python "$RUN_EVAL" --model "$MODEL" --case "$c" --rate "$rate" --wbits "$wbits" \
          2> "logs/ablation_${label}.log" | tr '\n' ' ')
    echo "$label | $cfg | $out" | tee -a logs/ablation.txt
}

python "$RUN_EVAL" --model "$MODEL" 2> logs/ablation_base.log | tr '\n' ' ' | sed 's/^/base | /' | tee -a logs/ablation.txt; echo

# uniform precision (no mixed-precision allocation): RTN vs GPTQ, with / without RHT
run c1_rtn_w4g128      1 0 4 '{"specs":["w4g128a"],"method":"rtn","rht":false}'
run c1_gptq_w4g128     1 0 4 '{"specs":["w4g128a"],"rht":false}'
run c1_gptq_rht_w4g128 1 0 4 '{"specs":["w4g128a"]}'
run c2_gptq_w3g128     2 0 3 '{"specs":["w3g128a"],"rht":false}'
run c2_gptq_rht_w3g128 2 0 3 '{"specs":["w3g128a"]}'
run c2_gptq_rht_w3ch   2 0 3 '{"specs":["w3cha"]}'
# mixed precision without RHT / without 2-bit candidates
run c2_mixed_norht     2 0.811 3 '{"rht":false}'
run c2_mixed_no2bit    2 0.811 3 '{"specs":["w4g128a","w4cha","w3g64a","w3g128a","w3g256a","w3cha"],"ref_specs":["w4g128a","w3g128a"]}'
# SVD: global vs uniform rank allocation, plain fp16 factors
run c3_uniform         3 0.5 8 '{"rank_alloc":"uniform"}'
run c3_svd_only_fp16   3 0 16 '{}'
# pruning ratio
run c4_prune025        4 0.851 3 '{"prune_ratio":0.25}'
run c4_prune_only      4 0 8 '{"specs":["w8cha"],"rht":false}'
