"""Weight quantization: group-wise uniform quantizer, GPTQ, randomized Hadamard transform (RHT).

All quantization is *simulated* ("fake quantization"): the dequantized weight is written back
into the nn.Linear in float16, while the storage it would need in a packed format
(integer codes + float16 scales + integer zero points + metadata) is computed exactly by
``QSpec.storage_bits`` and recorded with ``common.set_linear_bits``.
"""
import math
from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------------------
# quantization spec and storage cost
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QSpec:
    bits: int
    group_size: int = 128  # -1: one group per output row (per-channel)
    sym: bool = False      # asymmetric: integer zero point per group (stored with `bits` bits)
    scale_bits: int = 16   # float16 scales

    def groups(self, in_features):
        gs = in_features if self.group_size in (-1, 0) or self.group_size >= in_features else self.group_size
        return math.ceil(in_features / gs), gs

    def storage_bits(self, out_features, in_features):
        ng, _ = self.groups(in_features)
        per_group = self.scale_bits + (0 if self.sym else self.bits)
        return out_features * in_features * self.bits + out_features * ng * per_group

    def bits_per_weight(self, out_features, in_features):
        return self.storage_bits(out_features, in_features) / (out_features * in_features)

    def __str__(self):
        g = "ch" if self.group_size in (-1, 0) else f"g{self.group_size}"
        return f"w{self.bits}{g}{'s' if self.sym else 'a'}"


def parse_spec(s):
    """'w3g128a' / 'w4chs' -> QSpec."""
    assert s[0] == "w", s
    sym = s[-1] == "s"
    body = s[1:-1]
    if body.endswith("ch"):
        return QSpec(int(body[:-2]), -1, sym)
    b, g = body.split("g")
    return QSpec(int(b), int(g), sym)


# --------------------------------------------------------------------------------------
# uniform quantizer
# --------------------------------------------------------------------------------------
def find_params(w, spec, mse_grid=0, maxshrink=0.8, norm=2.4):
    """Scale / zero point for every row of ``w`` [rows, cols] (one group = one row here).

    With ``mse_grid > 0`` the clipping range is shrunk (grid search, as in the reference GPTQ
    implementation) to minimise the |w - q(w)|^norm error of each row.
    Returns (scale [rows, 1], zero [rows, 1]) in float32; scales are rounded to float16.
    """
    maxq = 2 ** spec.bits - 1
    w = w.float()
    xmin = w.min(1, keepdim=True)[0].clamp(max=0)
    xmax = w.max(1, keepdim=True)[0].clamp(min=0)
    if spec.sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = -xmax
    tiny = (xmax - xmin) < 1e-9
    xmin = torch.where(tiny, -torch.ones_like(xmin) * 1e-5, xmin)
    xmax = torch.where(tiny, torch.ones_like(xmax) * 1e-5, xmax)

    def params(lo, hi):
        scale = ((hi - lo) / maxq).half().float().clamp(min=1e-8)
        zero = torch.full_like(scale, (maxq + 1) / 2) if spec.sym else torch.round(-lo / scale)
        return scale, zero

    scale, zero = params(xmin, xmax)
    if mse_grid > 0:
        best = torch.full((w.shape[0], 1), float("inf"), device=w.device)
        for i in range(int(maxshrink * mse_grid)):
            p = 1 - i / mse_grid
            s1, z1 = params(p * xmin, p * xmax)
            err = (quant_dequant(w, s1, z1, maxq) - w).abs().pow(norm).sum(1, keepdim=True)
            better = err < best
            best = torch.where(better, err, best)
            scale = torch.where(better, s1, scale)
            zero = torch.where(better, z1, zero)
    return scale, zero


def quant_dequant(w, scale, zero, maxq):
    q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def rtn_quantize(W, spec, mse_grid=0):
    """Round-to-nearest group-wise quantization of W [out, in]; returns the dequantized weight."""
    out_f, in_f = W.shape
    ng, gs = spec.groups(in_f)
    W = W.float()
    pad = ng * gs - in_f
    Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
    Wg = Wp.reshape(out_f * ng, gs)
    scale, zero = find_params(Wg, spec, mse_grid)
    Q = quant_dequant(Wg, scale, zero, 2 ** spec.bits - 1).reshape(out_f, ng * gs)
    return Q[:, :in_f]


# --------------------------------------------------------------------------------------
# GPTQ (Frantar et al., 2022), with act-order + static groups
# --------------------------------------------------------------------------------------
@torch.no_grad()
def gptq_quantize(W, H, spec, blocksize=128, percdamp=0.01, actorder=True, mse_grid=0):
    """Quantize W [out, in] with GPTQ given the input Hessian H [in, in] = 2/n sum x x^T.

    Groups are always contiguous blocks of the original input columns ("static groups"), so
    act-order needs no per-column group index. Returns (dequantized W, proxy loss).
    """
    W = W.clone().float()
    H = H.clone().float()
    out_f, n = W.shape
    ng, gs = spec.groups(n)
    maxq = 2 ** spec.bits - 1

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # static group parameters computed on the original weights
    scales = torch.empty(out_f, ng, device=W.device)
    zeros = torch.empty(out_f, ng, device=W.device)
    for g in range(ng):
        s, z = find_params(W[:, g * gs:(g + 1) * gs], spec, mse_grid)
        scales[:, g], zeros[:, g] = s[:, 0], z[:, 0]
    col_group = torch.arange(n, device=W.device) // gs

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        col_group = col_group[perm]
        invperm = torch.argsort(perm)
    col_group = col_group.tolist()  # python ints: avoids a device sync per column

    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(n, device=W.device)
    for attempt in range(6):
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            break
        except torch.linalg.LinAlgError:
            H += damp * (10 ** attempt) * torch.eye(n, device=W.device)
    else:
        raise RuntimeError("GPTQ: Hessian is not positive definite")

    Q = torch.zeros_like(W)
    losses = torch.zeros(out_f, device=W.device)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            g = col_group[i1 + i]
            q = quant_dequant(w.unsqueeze(1), scales[:, g:g + 1], zeros[:, g:g + 1], maxq).flatten()
            Q1[:, i] = q
            losses += (w - q) ** 2 / d ** 2 / 2
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if actorder:
        Q = Q[:, invperm]
    return Q, losses.sum().item()


# --------------------------------------------------------------------------------------
# randomized Hadamard transform (incoherence processing, QuIP / QuIP# / QuaRot)
# --------------------------------------------------------------------------------------
_HAD_CACHE = {}


def hadamard_matrix(n, device):
    """Normalized Sylvester Hadamard matrix of size n (power of two)."""
    key = (n, str(device))
    if key not in _HAD_CACHE:
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HAD_CACHE[key] = (H / math.sqrt(n)).to(device)
    return _HAD_CACHE[key]


def largest_pow2_divisor(n, cap=4096):
    b = 1
    while n % (b * 2) == 0 and b * 2 <= cap:
        b *= 2
    return b


class RHT:
    """Orthogonal transform R = Hblk * diag(signs) * P on vectors of length n.

    P is a random permutation (mixes the Hadamard blocks when n is not a power of two),
    signs are random +-1, Hblk is block-diagonal with Sylvester-Hadamard blocks of the largest
    power-of-two size dividing n. Everything is generated from ``seed``; storing it costs the
    seed (64 bits) -- we conservatively also count the n sign bits.
    """

    def __init__(self, n, seed, device, cap=4096):
        g = torch.Generator().manual_seed(int(seed))
        self.n = n
        self.b = largest_pow2_divisor(n, cap)
        self.perm = torch.randperm(n, generator=g).to(device)
        self.inv_perm = torch.argsort(self.perm)
        self.signs = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float().to(device)
        self.Hb = hadamard_matrix(self.b, device)

    def storage_bits(self):
        return self.n + 64

    def forward(self, x):
        """R x along the last dimension."""
        shp = x.shape
        x = x[..., self.perm] * self.signs
        return (x.reshape(-1, self.n // self.b, self.b) @ self.Hb).reshape(shp)

    def inverse(self, x):
        """R^T x along the last dimension."""
        shp = x.shape
        x = (x.reshape(-1, self.n // self.b, self.b) @ self.Hb).reshape(shp)  # Hb is symmetric
        x = x * self.signs
        return x[..., self.inv_perm]


def rht_seed(base_seed, layer_idx, name, side):
    return (base_seed * 1000003 + layer_idx * 7919 + sum(map(ord, name)) * 31 + (1 if side == "in" else 2)) % (2 ** 62)


class IncoherentLinear:
    """Helper that maps W to W' = R_out W R_in^T and back (no-ops when disabled)."""

    def __init__(self, out_f, in_f, layer_idx, name, seed, device, enabled=True):
        self.enabled = enabled
        if enabled:
            self.rin = RHT(in_f, rht_seed(seed, layer_idx, name, "in"), device)
            self.rout = RHT(out_f, rht_seed(seed, layer_idx, name, "out"), device)

    def storage_bits(self):
        return self.rin.storage_bits() + self.rout.storage_bits() if self.enabled else 0

    def to_rotated(self, W):
        if not self.enabled:
            return W
        W = self.rin.forward(W.float())             # rows: w -> R_in w   (W R_in^T)
        return self.rout.forward(W.t()).t()          # cols: R_out W

    def from_rotated(self, Wr):
        if not self.enabled:
            return Wr
        W = self.rout.inverse(Wr.float().t()).t()
        return self.rin.inverse(W)

    def rotate_hessian(self, H):
        if not self.enabled:
            return H
        H = self.rin.forward(H.float())              # H R_in^T
        return self.rin.forward(H.t()).t()           # R_in H R_in^T

    def input_transform(self):
        return self.rin.forward if self.enabled else None
