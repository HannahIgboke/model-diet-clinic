"""The scales and the check-up: everything we measure about a patient."""

from __future__ import annotations

import math
import statistics
import time

import torch
import torch.nn.functional as F

from . import config


def device_of(model) -> torch.device:
    return next(model.parameters()).device


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Weight
# ---------------------------------------------------------------------------

def _tensors(model):
    yield from model.parameters()
    yield from model.buffers()


def footprint_bytes(model) -> int:
    """Bytes the model's weights occupy in memory.

    Shared tensors (like tied input/output embeddings) are counted once, and
    the small scale tables that 4-bit bitsandbytes layers keep alongside their
    packed weights are included.
    """
    seen, total = set(), 0
    for t in _tensors(model):
        if t.numel() == 0:
            continue
        key = t.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
        qs = getattr(t, "quant_state", None)
        if qs is not None:
            for extra in (getattr(qs, "absmax", None), getattr(getattr(qs, "state2", None), "absmax", None)):
                if isinstance(extra, torch.Tensor) and extra.data_ptr() not in seen:
                    seen.add(extra.data_ptr())
                    total += extra.numel() * extra.element_size()
    return total


def logical_params(model) -> int:
    """How many numbers the model is made of, regardless of how they're stored.

    A 4-bit layer packs two numbers into each byte, so we read its original
    shape instead of counting stored elements.
    """
    from .quantize import RefQuantLinear

    seen, n = set(), 0
    for mod in model.modules():
        if isinstance(mod, RefQuantLinear):
            n += mod.in_features * mod.out_features
        for p in mod.parameters(recurse=False):
            if p.data_ptr() in seen:
                continue
            seen.add(p.data_ptr())
            qs = getattr(p, "quant_state", None)
            shape = getattr(qs, "shape", None) if qs is not None else None
            n += math.prod(shape) if shape is not None else p.numel()
    return n


def nonzero_fraction(model) -> float:
    """Share of weights in linear layers that are not exactly zero."""
    total = nz = 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.is_floating_point():
            total += w.numel()
            nz += int(torch.count_nonzero(w).item())
    return nz / total if total else 1.0


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def make_chunks(tokenizer, text: str, max_tokens: int, chunk: int, max_positions: int | None):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0][:max_tokens]
    if max_positions:
        chunk = min(chunk, max_positions)
    chunks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    return [c for c in chunks if len(c) >= 16]


@torch.no_grad()
def _logits(model, chunk):
    x = chunk.unsqueeze(0).to(device_of(model))
    return model(input_ids=x, use_cache=False).logits[0].float()


@torch.no_grad()
def perplexity(model, chunks) -> float:
    nll, count = 0.0, 0
    for c in chunks:
        logits = _logits(model, c)
        target = c.to(logits.device)[1:]
        nll += F.cross_entropy(logits[:-1], target, reduction="sum").item()
        count += len(target)
    return math.exp(nll / max(count, 1))


@torch.no_grad()
def compare(original, dieted, chunks) -> dict:
    """How differently does the dieted model think, token by token?

    - perplexity: how surprised each model is by real text (lower is better)
    - agreement: how often both pick the same most-likely next token
    - kl: how far apart their whole probability spreads are (0 means identical)
    """
    nll_o = nll_d = kl_sum = agree = count = 0.0
    for c in chunks:
        lo = _logits(original, c)
        ld = _logits(dieted, c).to(lo.device)
        target = c.to(lo.device)[1:]
        nll_o += F.cross_entropy(lo[:-1], target, reduction="sum").item()
        nll_d += F.cross_entropy(ld[:-1], target, reduction="sum").item()
        lpo, lpd = F.log_softmax(lo, -1), F.log_softmax(ld, -1)
        kl_sum += (lpo.exp() * (lpo - lpd)).sum(-1).sum().item()
        agree += (lo.argmax(-1) == ld.argmax(-1)).sum().item()
        count += lo.shape[0]
        del lo, ld, lpo, lpd
    n_pred = count - len(chunks)
    return {
        "perplexity_before": math.exp(nll_o / max(n_pred, 1)),
        "perplexity_after": math.exp(nll_d / max(n_pred, 1)),
        "kl": kl_sum / max(count, 1),
        "agreement": agree / max(count, 1),
        "tokens_compared": int(count),
    }


def verdict(health: dict) -> dict:
    ratio = health["perplexity_after"] / max(health["perplexity_before"], 1e-9)
    agree = health["agreement"]
    if ratio < 1.10 and agree >= 0.85:
        level, text = "healthy", "The model still thinks almost exactly like itself."
    elif ratio < 1.6 and agree >= 0.6:
        level, text = "mild", "Noticeably lighter, a little foggier."
    else:
        level, text = "crash", "It lost the weight, but it isn't quite itself any more."
    return {"level": level, "text": text, "perplexity_ratio": ratio}


# ---------------------------------------------------------------------------
# Fitness and personality: side-by-side answers, with speed
# ---------------------------------------------------------------------------

CHAT_PROMPTS = [
    "Explain what it means to quantize a neural network, in two sentences.",
    "Write a short Python function that checks whether a word is a palindrome.",
    "Give me one fun fact about Lagos, Nigeria.",
]
BASE_PROMPTS = [
    "Quantizing a neural network means",
    "def is_palindrome(word):",
    "Lagos, the largest city in Nigeria, is",
]


def uses_chat(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def build_prompts(tokenizer, custom: str | None) -> list[str]:
    prompts = list(CHAT_PROMPTS if uses_chat(tokenizer) else BASE_PROMPTS)
    if custom and custom.strip():
        prompts = [custom.strip()[:500]] + prompts[:2]
    return prompts


def _encode(tokenizer, prompt: str, device) -> dict:
    if uses_chat(tokenizer):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
    # Only pass what every causal LM accepts (some tokenizers add token_type_ids).
    return {k: enc[k].to(device) for k in ("input_ids", "attention_mask") if k in enc}


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=None) -> list[dict]:
    max_new_tokens = max_new_tokens or config.SAMPLE_NEW_TOKENS
    dev = device_of(model)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs = dict(do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=pad)

    # Warm-up so the first timed answer isn't paying one-off start-up costs.
    warm = _encode(tokenizer, prompts[0], dev)
    model.generate(**warm, max_new_tokens=2, **gen_kwargs)

    out = []
    for p in prompts:
        enc = _encode(tokenizer, p, dev)
        _sync(dev)
        t0 = time.perf_counter()
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, **gen_kwargs)
        _sync(dev)
        dt = time.perf_counter() - t0
        new = ids[0, enc["input_ids"].shape[1]:]
        out.append({
            "prompt": p,
            "text": tokenizer.decode(new, skip_special_tokens=True).strip(),
            "tokens": int(len(new)),
            "seconds": dt,
            "tokens_per_sec": len(new) / dt if dt > 0 else 0.0,
        })
    return out


def median_speed(samples: list[dict]) -> float:
    speeds = [s["tokens_per_sec"] for s in samples if s["tokens"] > 0]
    return statistics.median(speeds) if speeds else 0.0


def peak_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None
