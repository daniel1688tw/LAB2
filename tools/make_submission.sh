#!/usr/bin/env bash
# Package the submission:  bash tools/make_submission.sh <StudentID>
# -> dist/<StudentID>.zip with Part1/<ID>_Part1.ipynb, Part2/<ID>_run.sh + run.sh ..., Part3/README.md ...
# The PDF reports may be named <ID>_report.pdf or StudentID_report.pdf (Part2/ and Part3/).
set -euo pipefail
ID=${1:?usage: make_submission.sh <StudentID>}
cd "$(dirname "$0")/.."
OUT=dist/$ID
rm -rf "$OUT" "dist/$ID.zip"
mkdir -p "$OUT/Part1" "$OUT/Part2/tests" "$OUT/Part3/tests"

cp Part1/StudentID_Part1.ipynb "$OUT/Part1/${ID}_Part1.ipynb"

cp Part2/{requirements.txt,run.sh,run_eval.py,evaluate.py,common.py,quant.py,pipeline.py,alloc.py,recipes.py,svd.py,prune.py} "$OUT/Part2/"
cp Part2/case{1,2,3,4}.py Part2/{ablation.sh,summarize_logs.py,README.md} "$OUT/Part2/"
cp Part2/tests/*.py Part2/tests/env.sh "$OUT/Part2/tests/"
sed "s/StudentID/$ID/g" Part2/StudentID_run.sh > "$OUT/Part2/${ID}_run.sh"
pdf=$(ls Part2/"${ID}"_report.pdf Part2/StudentID_report.pdf 2>/dev/null | head -1 || true)
[ -n "$pdf" ] && cp "$pdf" "$OUT/Part2/${ID}_report.pdf" || { echo "ERROR: Part2/${ID}_report.pdf missing" >&2; exit 1; }

cp Part3/{README.md,requirements.txt,run.sh,run_ptq.py,adaptquant.py,mxformats.py,evaluate.py,sweep.sh} "$OUT/Part3/"
cp Part3/tests/*.py "$OUT/Part3/tests/"
pdf=$(ls Part3/"${ID}"_report.pdf Part3/StudentID_report.pdf 2>/dev/null | head -1 || true)
[ -n "$pdf" ] && cp "$pdf" "$OUT/Part3/${ID}_report.pdf" || { echo "ERROR: Part3/${ID}_report.pdf missing" >&2; exit 1; }
sed -i "s/StudentID/$ID/g" "$OUT/Part3/README.md" "$OUT/Part2/README.md"

(cd "$OUT" && zip -qr "../$ID.zip" Part1 Part2 Part3)
echo "created dist/$ID.zip"; (cd dist && unzip -l "$ID.zip" | tail -n +4 | head -n -2 | awk '{print $4}')
