"""Run the unmodified run_eval.main() on CPU (test only: load_model's default device -> cpu).

usage: python tests/run_eval_cpu.py --model <dir> [--case N --rate R ...]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import evaluate  # noqa: E402

evaluate.load_model.__defaults__ = (evaluate.DEFAULT_MODEL, "float16", "cpu")
import run_eval  # noqa: E402

run_eval.main()
