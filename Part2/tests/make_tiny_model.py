"""Build an offline test bed: a tiny Qwen3 trained on Python-stdlib text, a BPE tokenizer and a fake
WikiText-2 cache (arrow files) so that the unmodified evaluate.py fallback path is exercised.

usage: python tests/make_tiny_model.py OUT_DIR [--steps 400]
"""
import argparse
import glob
import os
import random
import sysconfig

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--layers", type=int, default=6)
args = ap.parse_args()
torch.manual_seed(0)
random.seed(0)

files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**", "*.py"), recursive=True))
files = [f for f in files if "site-packages" not in f and "/test" not in f]
random.shuffle(files)
docs = [open(f, errors="ignore").read() for f in files[:600]]
train_docs, test_docs, calib_docs = docs[:480], docs[480:520], docs[520:]
assert test_docs and calib_docs, len(docs)
os.makedirs(args.out, exist_ok=True)

tk = Tokenizer(models.BPE(unk_token=None))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(train_docs, trainers.BpeTrainer(vocab_size=1024, special_tokens=["<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>")
model_dir = os.path.join(args.out, "tiny-qwen3")
tok.save_pretrained(model_dir)

cfg = Qwen3Config(vocab_size=1024, hidden_size=256, intermediate_size=768, num_hidden_layers=args.layers,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=2048,
                  tie_word_embeddings=True, rope_theta=10000.0)
model = Qwen3ForCausalLM(cfg)
ids = tok("\n\n".join(train_docs), return_tensors="pt").input_ids[0]
print("train tokens", ids.numel())
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.steps, pct_start=0.1)
L, B = 256, 16
model.train()
for step in range(args.steps):
    starts = torch.randint(0, ids.numel() - L - 1, (B,))
    x = torch.stack([ids[s:s + L] for s in starts])
    loss = model(x, labels=x).loss
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 50 == 0 or step == args.steps - 1:
        print(f"step {step} loss {loss.item():.3f}", flush=True)
model.eval()
model.save_pretrained(model_dir)

# fake WikiText-2 arrow cache: <cache>/wikitext/wikitext-2-raw-v1/0.0.0/x/wikitext-{split}.arrow
cache = os.path.join(args.out, "hf_datasets_cache")
for split, d in (("test", test_docs), ("train", calib_docs)):
    tmp = os.path.join(args.out, f"tmp_{split}")
    Dataset.from_dict({"text": [l for doc in d for l in doc.split("\n")]}).save_to_disk(tmp)
    dst = os.path.join(cache, "wikitext", "wikitext-2-raw-v1", "0.0.0", "deadbeef")
    os.makedirs(dst, exist_ok=True)
    os.replace(glob.glob(os.path.join(tmp, "*.arrow"))[0], os.path.join(dst, f"wikitext-{split}.arrow"))
with open(os.path.join(args.out, "calib.txt"), "w") as f:
    f.write("\n\n".join(calib_docs))
print("saved to", args.out)
