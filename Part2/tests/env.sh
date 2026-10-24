# source tests/env.sh <testbed dir> : offline environment for the end-to-end tests
export TB="$1"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="$TB/hf_datasets_cache"
export LAB2_CALIB_TEXT_FILE="$TB/calib.txt"
