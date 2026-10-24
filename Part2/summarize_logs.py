"""Turn logs/caseN.log into markdown tables for the report.   python summarize_logs.py [logs]"""
import glob
import os
import re
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "logs"
for path in sorted(glob.glob(os.path.join(d, "case*.log"))):
    text = open(path).read()
    print(f"\n### {os.path.basename(path)}\n")
    rows = re.findall(r"\]\s+(\w+_proj)\s+([\d.]+) MiB /\s+([\d.]+) MiB \(\s*([\d.]+) bits per original weight\)", text)
    if rows:
        print("| layer type | compressed (MiB) | original (MiB) | bits / original weight |")
        print("|---|---:|---:|---:|")
        for t, c, o, b in rows[-7:]:
            print(f"| {t} | {c} | {o} | {b} |")
    for pat in (r"compression rate = ([\d.]+%)", r"SVD compression rate[^:]*: ([\d.]+%)",
                r"pruning rate \(parameters removed\): ([\d.]+%)", r"allocation: (.*)", r"rank allocation: (.*)",
                r"pruning plan \(channels removed per layer\): (.*)"):
        m = re.findall(pat, text)
        if m:
            print(f"\n- `{pat.split('(')[0].strip()}` {m[-1]}")
    alloc = re.findall(r"\]\s{3}(\w+_proj)\s{2,}(w\d.*)", text)
    if alloc:
        print("\n| layer type | specs (count) |\n|---|---|")
        for t, a in alloc[:7]:
            print(f"| {t} | {a} |")
