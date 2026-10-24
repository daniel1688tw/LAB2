"""Run every code cell of the Part 1 notebook except pip installs and the MobileNetV2 (Kaggle file) cells."""
import json
import sys
import matplotlib
matplotlib.use("Agg")

nb = json.load(open(sys.argv[1]))
g = {}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "\n!" in "\n" + src or "mobilenetv2_0.963.pth" in src:
        print(f"[skip cell {i}]")
        continue
    exec(compile(src, f"cell{i}", "exec"), g)
print("ALL CELLS OK")
