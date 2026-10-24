"""Shared utilities for the Part 2 compression cases.

* storage accounting of the target Linear layers (``record_base`` / ``set_linear_bits`` /
  ``compressed_linear_bits``), used by ``evaluate.compression_rate``;
* calibration data (random windows of the WikiText-2 *train* split, never the test split);
* a layer-by-layer ("sequential") driver that feeds calibration activations through the
  decoder blocks one at a time, which is what GPTQ / SVD-LLM / OBS pruning need.

Target layers = the 7 Linear layers of every decoder block
(q/k/v/o_proj of self-attention and gate/up/down_proj of the MLP).
The embedding / tied lm_head is NOT a target layer and is never modified.
"""
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn

TARGET_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# GPTQ "true-sequential" order inside a block: each group sees the outputs of the already
# compressed groups before it.
SEQUENTIAL_GROUPS = (("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",))
BASE_BITS = 16  # the reference model is evaluated in float16


def log(*args):
    """Logs go to stderr so that run_eval.py's stdout only carries the final result."""
    print(f"[{time.strftime('%H:%M:%S')}]", *args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------
# configuration: every case has a DEFAULTS dict that can be overridden from the environment
# --------------------------------------------------------------------------------------
def get_config(case_name, defaults):
    """``LAB2_<CASE>_CFG='{"nsamples": 256}'`` overrides individual keys of ``defaults``."""
    cfg = dict(defaults)
    raw = os.environ.get(f"LAB2_{case_name.upper()}_CFG")
    if raw:
        override = json.loads(raw)
        unknown = set(override) - set(cfg)
        if unknown:
            raise KeyError(f"unknown config keys for {case_name}: {sorted(unknown)}")
        cfg.update(override)
    return cfg


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# model structure helpers
# --------------------------------------------------------------------------------------
def get_layers(model):
    return model.model.layers


def block_linears(layer):
    """{type: nn.Linear} of one decoder block."""
    return {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }


def target_linears(model):
    """{'layers.<i>.<type>': nn.Linear} for all target layers, in model order."""
    out = {}
    for i, layer in enumerate(get_layers(model)):
        for t, m in block_linears(layer).items():
            out[f"layers.{i}.{t}"] = m
    return out


def model_device(model):
    return next(model.parameters()).device


# --------------------------------------------------------------------------------------
# storage accounting
# --------------------------------------------------------------------------------------
def record_base(model):
    """Remember the size of the uncompressed target layers. Must be called before compressing."""
    lin = target_linears(model)
    model._base_linear_numel = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0) for m in lin.values())
    model._base_linear_shapes = {n: tuple(m.weight.shape) for n, m in lin.items()}
    model._linear_bits = {}   # name -> storage bits of the compressed representation
    model._linear_notes = {}  # name -> human readable description
    model._extra_stats = {}


def set_linear_bits(model, name, bits, note=""):
    model._linear_bits[name] = float(bits)
    if note:
        model._linear_notes[name] = note


def compressed_linear_bits(model):
    """Total storage (bits) of the target layers.

    Layers that were explicitly compressed report their own size (codes + scales + zero points +
    any other metadata); untouched layers count as float16 with their *current* shape, so a layer
    that was structurally pruned but not quantized is still accounted correctly.
    """
    total = 0.0
    for name, m in target_linears(model).items():
        if name in model._linear_bits:
            total += model._linear_bits[name]
        else:
            total += BASE_BITS * (m.weight.numel() + (m.bias.numel() if m.bias is not None else 0))
    return total


def compression_rate(model):
    if not hasattr(model, "_base_linear_numel"):
        return 0.0
    return 1.0 - compressed_linear_bits(model) / (BASE_BITS * model._base_linear_numel)


def report(model, title):
    """Per-layer-type summary of the storage, printed to stderr."""
    lin = target_linears(model)
    base = {t: 0 for t in TARGET_TYPES}
    comp = {t: 0.0 for t in TARGET_TYPES}
    for name, m in lin.items():
        t = name.split(".")[-1]
        base[t] += BASE_BITS * math.prod(model._base_linear_shapes[name])
        comp[t] += model._linear_bits.get(name, BASE_BITS * m.weight.numel())
    log(f"==== {title}: storage of the target Linear layers ====")
    for t in TARGET_TYPES:
        log(f"  {t:10s} {comp[t] / 8 / 2**20:9.1f} MiB / {base[t] / 8 / 2**20:9.1f} MiB "
            f"({comp[t] / base[t] * 16:5.2f} bits per original weight)")
    for k, v in model._extra_stats.items():
        log(f"  {k}: {v}")
    log(f"  compression rate = {compression_rate(model):.2%}")


# --------------------------------------------------------------------------------------
# calibration data
# --------------------------------------------------------------------------------------
def _wikitext2_train_text():
    from datasets import Dataset, load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except (PermissionError, OSError, ConnectionError, ValueError):
        cache = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
        ds = Dataset.from_file(sorted(glob.glob(os.path.join(cache, "wikitext/wikitext-2-raw-v1/*/*/wikitext-train.arrow")))[0])
    return "\n\n".join(ds["text"])


def _c4_text(n_docs=3000, seed=0):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
    idx = random.Random(seed).sample(range(len(ds)), n_docs)
    return "\n\n".join(ds[i]["text"] for i in idx)


def get_calib_data(tokenizer, nsamples=128, seqlen=2048, seed=0, dataset="wikitext2"):
    """``nsamples`` random windows of ``seqlen`` tokens -> LongTensor [nsamples, seqlen] (CPU).

    ``dataset='synthetic'`` (or env LAB2_SYNTHETIC_CALIB=1) returns random token ids; only used by
    the unit tests that run without internet access.
    """
    if dataset == "synthetic" or os.environ.get("LAB2_SYNTHETIC_CALIB") == "1":
        g = torch.Generator().manual_seed(seed)
        vocab = int(os.environ.get("LAB2_SYNTHETIC_VOCAB", "1000"))
        return torch.randint(0, vocab, (nsamples, seqlen), generator=g)
    if os.environ.get("LAB2_CALIB_TEXT_FILE"):  # offline tests
        text = open(os.environ["LAB2_CALIB_TEXT_FILE"]).read()
    else:
        text = _wikitext2_train_text() if dataset == "wikitext2" else _c4_text(seed=seed)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    rng = random.Random(seed)
    starts = [rng.randint(0, ids.numel() - seqlen - 1) for _ in range(nsamples)]
    return torch.stack([ids[s:s + seqlen] for s in starts])


# --------------------------------------------------------------------------------------
# sequential (block-by-block) driver
# --------------------------------------------------------------------------------------
class _Catcher(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.inputs, self.kwargs = [], None

    def __getattr__(self, name):  # keep attributes such as `attention_type` reachable
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)

    def forward(self, hidden_states, *args, **kwargs):
        self.inputs.append(hidden_states.detach())
        self.kwargs = kwargs
        raise ValueError("catcher")


@torch.no_grad()
def capture_block_inputs(model, calib_ids, dev=None):
    """Run the embedding only and return (inputs of block 0 [N, L, H], kwargs of the block call)."""
    dev = dev or model_device(model)
    layers = get_layers(model)
    catcher = _Catcher(layers[0])
    layers[0] = catcher
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        for i in range(calib_ids.shape[0]):
            try:
                model(calib_ids[i:i + 1].to(dev))
            except ValueError:
                pass
    finally:
        layers[0] = catcher.layer
        model.config.use_cache = use_cache
    inps = torch.cat(catcher.inputs, 0)
    kwargs = {k: v for k, v in catcher.kwargs.items() if k not in ("past_key_value", "past_key_values", "use_cache")}
    kwargs["use_cache"] = False
    return inps, kwargs


@torch.no_grad()
def run_block(layer, inps, kwargs, bs=1):
    """Forward [N, L, H] activations through one decoder block (batch of ``bs`` sequences)."""
    outs = torch.empty_like(inps)
    for i in range(0, inps.shape[0], bs):
        out = layer(inps[i:i + bs], **kwargs)
        outs[i:i + bs] = out[0] if isinstance(out, (tuple, list)) else out
    return outs


class InputStats:
    """Forward hook that accumulates H = (2/n) * sum x x^T of a Linear's input (GPTQ convention).

    ``transform`` (optional) maps the input before accumulation (e.g. a Hadamard rotation).
    ``diag_only`` keeps only diag(H) (cheap sensitivity statistics).
    """

    def __init__(self, module, transform=None, diag_only=False):
        n = module.in_features
        dev = module.weight.device
        self.diag_only = diag_only
        self.H = torch.zeros(n, device=dev) if diag_only else torch.zeros(n, n, device=dev)
        self.nsamples = 0
        self.transform = transform
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        x = inp[0].detach()
        x = x.reshape(-1, x.shape[-1]).float()
        if self.transform is not None:
            x = self.transform(x)
        n = x.shape[0]
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        x = x * math.sqrt(2 / self.nsamples)
        if self.diag_only:
            self.H += (x * x).sum(0)
        else:
            self.H += x.t() @ x

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def collect_stats(layer, names, inps, kwargs, transform=None, diag_only=False, bs=1):
    """Run the block on ``inps`` and return {type: InputStats} for the given Linear types."""
    lin = block_linears(layer)
    stats = {t: InputStats(lin[t], transform(t) if transform else None, diag_only) for t in names}
    try:
        run_block(layer, inps, kwargs, bs)
    finally:
        for s in stats.values():
            s.remove()
    return stats


def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
