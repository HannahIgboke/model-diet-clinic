"""Diet 1: Quantization, or "smaller portions".

Every weight is kept, but stored with fewer bits. On a GPU the clinic uses
bitsandbytes, the library most people use in practice. Without a GPU (or if
you pass backend="reference") it uses the small reference implementation
below, which shows the whole trick in about 60 lines.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

PLANS = {
    "bf16": {"label": "Half portions (16-bit)", "bits": 16},
    "int8": {"label": "Small portions (8-bit)", "bits": 8},
    "nf4": {"label": "Tiny portions (4-bit)", "bits": 4},
}


def _conv1d_cls():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except Exception:  # pragma: no cover
        return ()


class RefQuantLinear(nn.Module):
    """A linear layer whose weights are stored as 8-bit or 4-bit integers.

    8-bit: one scale per output row; weights rounded to -127..127.
    4-bit: one scale per group of `group_size` weights; rounded to -8..7 and
    packed two to a byte. Weights are expanded back just in time for each
    forward pass, which is slow but makes the memory saving real and visible.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, bits: int, group_size: int = 64):
        super().__init__()
        W = weight.detach().float()
        self.out_features, self.in_features = W.shape
        self.bits = bits
        self.compute_dtype = weight.dtype
        if bits == 8:
            scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            q = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
            self.group_size = self.in_features
        elif bits == 4:
            gs = group_size
            pad = (-self.in_features) % gs
            if pad:
                W = F.pad(W, (0, pad))
            Wg = W.view(self.out_features, -1, gs)
            scale = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
            q = (torch.round(Wg / scale).clamp(-8, 7) + 8).to(torch.uint8).view(self.out_features, -1)
            q = q[:, 0::2] | (q[:, 1::2] << 4)          # two 4-bit numbers per byte
            scale = scale.squeeze(-1)
            self.group_size = gs
        else:
            raise ValueError(f"bits must be 8 or 4, got {bits}")
        self.register_buffer("qweight", q.contiguous())
        self.register_buffer("scale", scale.to(torch.float16).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def dequantize(self) -> torch.Tensor:
        s = self.scale.float()
        if self.bits == 8:
            return self.qweight.float() * s
        lo = (self.qweight & 0x0F).to(torch.int16) - 8
        hi = (self.qweight >> 4).to(torch.int16) - 8
        q = torch.stack([lo, hi], dim=-1).view(self.out_features, -1).float()
        W = (q.view(self.out_features, -1, self.group_size) * s.unsqueeze(-1)).view(self.out_features, -1)
        return W[:, : self.in_features]

    def forward(self, x):
        w = self.dequantize().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}"


def _skip(name: str, module, model) -> bool:
    # Like bitsandbytes, leave the output head at full precision: it is the
    # layer most sensitive to rounding.
    if name.endswith("lm_head"):
        return True
    out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    return out is not None and module is out


def reference_quantize(model, bits: int):
    """Swap every linear layer (except the output head) for RefQuantLinear."""
    Conv1D = _conv1d_cls()
    targets = []
    for name, mod in model.named_modules():
        if _skip(name, mod, model):
            continue
        if isinstance(mod, nn.Linear):
            targets.append((name, mod.weight, mod.bias))
        elif Conv1D and isinstance(mod, Conv1D):      # GPT-2 style: weight is (in, out)
            targets.append((name, mod.weight.t(), mod.bias))
    for name, w, b in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, RefQuantLinear(w, b, bits))
    return len(targets)


def bnb_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def quantize(original, load_fn, plan: str, backend: str = "auto", half_dtype=torch.bfloat16):
    """Return (dieted_model, extras).

    `load_fn(**kwargs)` reloads the patient from disk with extra
    `from_pretrained` arguments; bitsandbytes quantizes while loading.
    """
    if plan not in PLANS:
        raise ValueError(f"Unknown quantization plan '{plan}'")
    extras = {"plan": plan, "label": PLANS[plan]["label"], "bits": PLANS[plan]["bits"]}

    if plan == "bf16":
        dieted = copy.deepcopy(original).to(half_dtype)
        extras["backend"] = "torch"
        if half_dtype != torch.bfloat16:
            extras["note"] = "This GPU has no bfloat16 support, so float16 was used (same size)."
        return dieted, extras

    use_bnb = backend == "bitsandbytes" or (backend == "auto" and bnb_available())
    if use_bnb:
        from transformers import BitsAndBytesConfig

        if plan == "int8":
            qc = BitsAndBytesConfig(load_in_8bit=True)
        else:
            qc = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=half_dtype,
            )
        dieted = load_fn(quantization_config=qc)
        extras["backend"] = "bitsandbytes"
        return dieted, extras

    dieted = copy.deepcopy(original)
    n = reference_quantize(dieted, PLANS[plan]["bits"])
    extras["backend"] = "reference"
    extras["layers_quantized"] = n
    if plan == "nf4":
        extras["note"] = "The reference backend uses plain 4-bit integers (groups of 64), not NF4."
    return dieted, extras
