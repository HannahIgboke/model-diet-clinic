"""The treatment room: weigh the patient, apply a diet, weigh again, check up."""

from __future__ import annotations

import gc
import os
import time
import uuid
from datetime import datetime, timezone

import torch

from . import config
from . import measure as M
from .menu import DIETS, validate

DATA = os.path.join(os.path.dirname(__file__), "data", "clinic_text.txt")

def _texts():
    with open(DATA, encoding="utf-8") as f:
        train, _, evaluation = f.read().partition("=== EVAL ===")
    return train.strip(), evaluation.strip()


def _dtype_kw():
    # Newer transformers renamed torch_dtype -> dtype.
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def _hardware(dev: torch.device) -> str:
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    return "CPU"


def _free(dev: torch.device | None = None):
    gc.collect()
    if dev is not None and dev.type == "cuda":
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def native_bf16(dev: torch.device) -> bool:
    """Older GPUs (like the T4 on Kaggle and Colab) can't do bfloat16 maths in hardware."""
    if dev.type != "cuda":
        return True
    return torch.cuda.get_device_capability(dev)[0] >= 8


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_dtype(model_dir: str, dev: torch.device):
    """Keep the model's own precision, except bfloat16 on GPUs without bf16 support,
    where float16 is used instead (same size, and fast on those cards)."""
    if native_bf16(dev):
        return "auto"
    try:
        import json
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        stored = str(cfg.get("torch_dtype") or cfg.get("dtype") or (cfg.get("text_config") or {}).get("torch_dtype") or "")
    except Exception:
        stored = ""
    return torch.float16 if "bfloat16" in stored else "auto"


def _weigh(model, samples) -> dict:
    b = M.footprint_bytes(model)
    p = M.logical_params(model)
    return {
        "bytes": b,
        "gb": b / 1024**3,
        "params": p,
        "bytes_per_param": b / p if p else None,
        "tokens_per_sec": M.median_speed(samples),
        "nonzero_fraction": M.nonzero_fraction(model),
    }


def treat(model_dir: str, patient: dict, diet: str, plan: str, options: dict | None = None,
          prompt: str | None = None, emit=lambda e: None, device: str | None = None) -> dict:
    """Run one full visit and return the weigh-in card as a dict."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    opts = validate(diet, plan, options)
    dev = pick_device(device)
    dkw = _dtype_kw()
    load_dtype = _load_dtype(model_dir, dev)
    half = torch.bfloat16 if native_bf16(dev) else torch.float16
    timings = {}

    def step(stage, message, pct=None):
        e = {"type": "progress", "stage": stage, "message": message}
        if pct is not None:
            e["pct"] = pct
        emit(e)

    def load(**kw):
        return AutoModelForCausalLM.from_pretrained(
            model_dir, **{dkw: load_dtype}, device_map=str(dev), low_cpu_mem_usage=True,
            trust_remote_code=False, **kw,
        ).eval()

    # 1. Weigh-in -----------------------------------------------------------
    step("weigh_in", "Helping the patient onto the scale (loading the model)…", 0.05)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    original = load()
    timings["load_seconds"] = time.perf_counter() - t0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    train_text, eval_text = _texts()
    max_pos = getattr(original.config, "max_position_embeddings", None)
    eval_chunks = M.make_chunks(tok, eval_text, config.EVAL_MAX_TOKENS, config.EVAL_CHUNK, max_pos)
    train_chunks = M.make_chunks(tok, train_text, 4096, 128, max_pos)

    prompts = M.build_prompts(tok, prompt)
    step("weigh_in", "Asking the patient a few questions before the diet…", 0.15)
    samples_before = M.generate(original, tok, prompts)
    before = _weigh(original, samples_before)
    before["dtype"] = str(next(original.parameters()).dtype).replace("torch.", "")
    before["load_seconds"] = timings["load_seconds"]
    emit({"type": "weighed", "which": "before", "bytes": before["bytes"], "params": before["params"]})

    # 2. The diet -------------------------------------------------------------
    label = DIETS[diet]["plans"][plan]
    step("diet", f"Starting the diet: {label}…", 0.3)
    t0 = time.perf_counter()
    if diet == "quantize":
        from .quantize import quantize
        _free(dev)
        dieted, extras = quantize(original, load, plan, half_dtype=half)
    elif diet == "prune":
        from .prune import layer_prune
        step("diet", "Looking for empty calories: measuring how hard each layer works…", 0.35)
        dieted, extras = layer_prune(original, eval_chunks, opts["fraction"])
    else:
        from .distill import distill
        step("diet", "Warming up the training partner…", 0.35)
        dieted, extras = distill(original, train_chunks, eval_chunks, keep_fraction=opts["keep_fraction"],
                                 steps=opts["steps"], emit=emit)
    dieted.eval()
    timings["diet_seconds"] = time.perf_counter() - t0
    _free(dev)

    # 3. Re-weigh and check-up ---------------------------------------------
    step("check_up", "Back on the scale…", 0.7)
    step("check_up", "Running the check-up: comparing how both versions think…", 0.75)
    health = M.compare(original, dieted, eval_chunks)
    health.update(M.verdict(health))
    step("check_up", "Asking the same questions again…", 0.88)
    samples_after = M.generate(dieted, tok, prompts)
    after = _weigh(dieted, samples_after)
    after["dtype"] = str(next(dieted.parameters()).dtype).replace("torch.", "")
    emit({"type": "weighed", "which": "after", "bytes": after["bytes"], "params": after["params"]})

    peak = M.peak_memory_bytes(dev)
    lost = before["bytes"] - after["bytes"]
    result = {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patient": patient,
        "diet": {"key": diet, "name": DIETS[diet]["name"], "technical": DIETS[diet]["technical"],
                 "tagline": DIETS[diet]["tagline"], "picture": DIETS[diet]["picture"],
                 "plan": plan, "label": label, "options": opts},
        "before": before,
        "after": after,
        "change": {
            "bytes_lost": lost,
            "gb_lost": lost / 1024**3,
            "percent_lost": 100 * lost / before["bytes"] if before["bytes"] else 0,
            "speedup": (after["tokens_per_sec"] / before["tokens_per_sec"]) if before["tokens_per_sec"] else None,
        },
        "health": health,
        "samples": [
            {"prompt": b["prompt"], "before": b["text"], "after": a["text"],
             "before_tps": b["tokens_per_sec"], "after_tps": a["tokens_per_sec"]}
            for b, a in zip(samples_before, samples_after)
        ],
        "extras": extras,
        "hardware": _hardware(dev),
        "peak_memory_bytes": peak,
        "timings": timings,
    }

    del original, dieted
    _free(dev)
    step("done", "All done!", 1.0)
    return result
