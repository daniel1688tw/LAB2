"""Fill the YOUR CODE blocks of lab2_template/part1.ipynb and write Part1/StudentID_Part1.ipynb."""
import json
import re
import sys

SRC, DST = sys.argv[1], sys.argv[2]

# (function signature that identifies the cell, list of (placeholder line, replacement)) in order
FILLS = {
    "def linear_quantize(": [
        ("    scaled_tensor = ...", "    scaled_tensor = fp_tensor / scale"),
        ("    rounded_tensor = ...", "    rounded_tensor = torch.round(scaled_tensor)"),
        ("    shifted_tensor = ...", "    shifted_tensor = rounded_tensor + zero_point"),
    ],
    "def get_quantization_scale_and_zero_point(": [
        ("    scale = ...", "    scale = (fp_max - fp_min) / (quantized_max - quantized_min)"),
        ("    zero_point = ...", "    zero_point = quantized_min - fp_min / scale"),
    ],
    "def get_quantization_scale_for_weight(": [
        ("    scale = ...", "    scale = fp_max / quantized_max"),
    ],
    "def linear_quantize_bias_per_output_channel(": [
        ("    bias_scale = ...", "    bias_scale = weight_scale * input_scale"),
    ],
    "def quantized_linear(": [
        ("    output = ...", "    output = output.float() * (input_scale * weight_scale / output_scale).view(1, -1)"),
        ("    output = ...", "    output = output + output_zero_point"),
    ],
    "def quantized_conv2d(": [
        ("    output = ...", "    output = output.float() * (input_scale * weight_scale / output_scale).view(1, -1, 1, 1)"),
        ("    output = ...", "    output = output + output_zero_point"),
    ],
    "def linear_quantize_weight_per_group(": [
        ("    grouped = ...", "    grouped = tensor.view(oc, ic // group_size, group_size)"),
        ("    fp_max = ...", "    fp_max = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=5e-7)"),
        ("    scale = ...", "    scale = fp_max / quantized_max"),
    ],
    "def search_clipping_range(": [
        ("        clip = ...", "        clip = ratio * fp_max"),
        ("        scale = ...", "        scale = clip / quantized_max"),
        ("        quantized = ...", "        quantized = linear_quantize(fp_tensor, bitwidth, scale, 0)"),
        ("        dequantized = ...", "        dequantized = scale * quantized.float()"),
        ("        mse = ...", "        mse = ((dequantized - fp_tensor) ** 2).mean().item()"),
    ],
    "def fold_conv_bn(": [
        ("    factor = ...", "    factor = bn_weight / torch.sqrt(running_var + eps)"),
        ("    fused_weight = ...", "    fused_weight = conv_weight * factor.view(-1, 1, 1, 1)"),
        ("    fused_bias = ...", "    fused_bias = (conv_bias - running_mean) * factor + bn_bias"),
    ],
    "class FakeQuantizeSTE(": [
        ("        quantized_tensor = ...", "        quantized_tensor = linear_quantize(fp_tensor, bitwidth, scale, zero_point)"),
        ("        dequantized_tensor = ...", "        dequantized_tensor = scale * (quantized_tensor.float() - zero_point)"),
        ("        r_min = ...", "        r_min = scale * (quantized_min - zero_point)"),
        ("        r_max = ...", "        r_max = scale * (quantized_max - zero_point)"),
        ("        mask = ...", "        mask = (fp_tensor >= r_min) & (fp_tensor <= r_max)"),
        ("        grad_input = ...", "        grad_input = grad_output * mask.to(grad_output.dtype)"),
    ],
}

nb = json.load(open(SRC))
done = set()
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    for key, repls in FILLS.items():
        if key in src and not src.lstrip().startswith("def test_"):
            for old, new in repls:
                assert old + "\n" in src, (key, old)
                src = src.replace(old + "\n", new + "\n", 1)
            assert "= ...\n" not in src, key
            done.add(key)
            cell["source"] = src.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
assert done == set(FILLS), set(FILLS) - done
json.dump(nb, open(DST, "w"), indent=1, ensure_ascii=False)
print("filled", len(done), "cells ->", DST)
