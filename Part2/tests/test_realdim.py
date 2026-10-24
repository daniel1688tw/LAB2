"""Shape test with the exact Qwen3-4B layer dimensions (hidden 2560, intermediate 9728, 32/8 heads,
head_dim 128), 3 random layers, float32 on CPU, synthetic calibration ids.
Checks that cases 2/3/4 run, hit their target compression rate and keep the model finite.
python tests/test_realdim.py
"""
import json
import os
import sys
import time

os.environ["LAB2_SYNTHETIC_CALIB"] = "1"
small = {"nsamples": 4, "seqlen": 128, "sens_nsamples": 2, "sens_seqlen": 64, "mse_grid": 0,
         "cache_dir": os.path.join(os.environ.get("TMPDIR", "/tmp"), "lab2_realdim_cache")}
os.environ["LAB2_CASE2_CFG"] = json.dumps(dict(small, specs=["w4cha", "w3g128a", "w3cha", "w2g128a"]))
os.environ["LAB2_CASE3_CFG"] = json.dumps({"nsamples": 4, "seqlen": 128})
os.environ["LAB2_CASE4_CFG"] = json.dumps(dict(small, specs=["w4cha", "w3g128a", "w3cha", "w2g128a"],
                                               skip_first=1, skip_last=0))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

import common  # noqa: E402

cfg = Qwen3Config(vocab_size=1000, hidden_size=2560, intermediate_size=9728, num_hidden_layers=3,
                  num_attention_heads=32, num_key_value_heads=8, head_dim=128, tie_word_embeddings=True)
x = torch.randint(0, 1000, (1, 64))
for case, rate, wbits in ((2, 0.811, 3), (4, 0.851, 3), (3, 0.5, 8)):
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    common.record_base(model)
    t0 = time.time()
    __import__(f"case{case}").compress(model, None, rate=rate, wbits=wbits, seed=0)
    r = common.compression_rate(model)
    with torch.no_grad():
        logits = model(x).logits
    assert torch.isfinite(logits).all()
    assert r >= rate, (case, r)
    print(f"case {case}: compression rate {r:.4%} (target {rate}), {time.time() - t0:.0f}s", flush=True)
print("REAL-DIM TEST PASSED")
