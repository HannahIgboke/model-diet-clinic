"""Check-in desk: turn whatever link a visitor pastes into a model we can treat.

Everything here runs on a cheap CPU machine and only reads metadata, so a
patient that is too big, gated or unsafe is turned away in a second, before a
single gigabyte is downloaded or a GPU is booked.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict

from . import config

GB = 1024**3


class PatientRejected(Exception):
    """Raised when a model can't be admitted. `code` is machine-readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Patient:
    repo_id: str            # e.g. "Qwen/Qwen2.5-0.5B-Instruct", or a local folder
    revision: str | None    # exact commit, so every visitor treats the same weights
    link: str
    size_bytes: int         # total size of the weight files
    params: int | None      # parameter count, when the Hub reports it
    dtype: str | None       # storage precision, e.g. "BF16"
    architecture: str
    model_type: str
    num_layers: int | None
    is_local: bool = False

    @property
    def size_gb(self) -> float:
        return self.size_bytes / GB

    def to_dict(self) -> dict:
        d = asdict(self)
        d["size_gb"] = round(self.size_gb, 3)
        d["allowed_diets"] = allowed_diets(self)
        return d


_HF_HOSTS = ("huggingface.co", "hf.co", "www.huggingface.co")
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def parse_link(link: str) -> str:
    """Accept a Hub URL or a bare 'org/name' and return 'org/name'."""
    link = (link or "").strip().strip("/")
    if not link:
        raise PatientRejected("empty", "Paste a model link first.")

    m = re.match(r"^(?:https?://)?([^/]+)/(.*)$", link)
    if m and "." in m.group(1):
        host, path = m.group(1).lower(), m.group(2)
        if host not in _HF_HOSTS:
            raise PatientRejected(
                "unsupported_host",
                f"The clinic only admits models hosted on Hugging Face for now ({host} isn't supported). "
                "Clone the repo to add other sources.",
            )
        parts = [p for p in path.split("/") if p]
        if parts and parts[0] in ("models",):
            parts = parts[1:]
        if parts and parts[0] in ("datasets", "spaces"):
            raise PatientRejected("not_a_model", "That link points to a dataset or app, not a model.")
        if len(parts) < 2:
            raise PatientRejected("bad_link", "That link doesn't look like a model page (expected huggingface.co/org/model).")
        repo = f"{parts[0]}/{parts[1]}"
    else:
        repo = link

    if not _REPO_RE.match(repo):
        raise PatientRejected("bad_link", "That doesn't look like a model link. Try something like huggingface.co/Qwen/Qwen2.5-0.5B-Instruct")
    return repo


def allowed_diets(p: Patient) -> dict:
    """Which diets this patient can go on, and why not if not."""
    out = {"quantize": {"ok": True}, "prune": {"ok": True}, "distill": {"ok": True}}
    if p.size_gb > config.MAX_DISTILL_GB:
        out["distill"] = {
            "ok": False,
            "reason": (
                f"Distillation trains a whole new model, so the clinic only offers it for patients up to "
                f"{config.MAX_DISTILL_GB:g}GB. This one is {p.size_gb:.1f}GB: clone the repo and run it on a bigger GPU."
            ),
        }
    if p.num_layers is not None and p.num_layers < 4:
        msg = "This model has too few layers to cut any out safely."
        out["prune"] = {"ok": False, "reason": msg}
        out["distill"] = {"ok": False, "reason": msg}
    return out


_WEIGHT_SUFFIXES = (".safetensors",)


def _check_config(cfg: dict) -> tuple[str, str, int | None]:
    if cfg.get("auto_map"):
        raise PatientRejected(
            "custom_code",
            "This model needs to run its own custom code to load. The shared clinic doesn't run strangers' code; "
            "clone the repo and run it yourself if you trust it.",
        )
    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""
    if not arch.endswith(("ForCausalLM", "LMHeadModel")):
        kind = arch or cfg.get("model_type") or "unknown"
        raise PatientRejected(
            "not_causal_lm",
            f"The clinic treats text-generation models (…ForCausalLM). This one is '{kind}'.",
        )
    text_cfg = cfg.get("text_config") or cfg
    layers = None
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if isinstance(text_cfg.get(key), int):
            layers = text_cfg[key]
            break
    return arch, cfg.get("model_type", ""), layers


def _check_size(size: int):
    gb = size / GB
    if gb > config.MAX_MODEL_GB:
        raise PatientRejected(
            "too_big",
            f"This patient weighs {gb:.1f}GB, and the clinic's scale only goes up to {config.MAX_MODEL_GB:g}GB. "
            "Clone the repo and raise DIET_MAX_MODEL_GB to treat it on your own GPU.",
        )


def inspect_local(path: str) -> Patient:
    """Check in a model from a local folder (local mode only)."""
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        raise PatientRejected("bad_link", f"No config.json found in {path}.")
    with open(cfg_path) as f:
        cfg = json.load(f)
    arch, model_type, layers = _check_config(cfg)
    files = [f for f in os.listdir(path) if f.endswith(_WEIGHT_SUFFIXES)]
    if not files:
        raise PatientRejected("no_safetensors", "No .safetensors weights found in that folder.")
    size = sum(os.path.getsize(os.path.join(path, f)) for f in files)
    _check_size(size)
    return Patient(
        repo_id=path, revision=None, link=path, size_bytes=size, params=None,
        dtype=cfg.get("torch_dtype") or cfg.get("dtype"), architecture=arch,
        model_type=model_type, num_layers=layers, is_local=True,
    )


def inspect_model(link: str, allow_local: bool = False, token: str | None = None) -> Patient:
    """Look a model up on the Hub and decide whether to admit it."""
    if allow_local and link and os.path.isdir(link.strip()):
        return inspect_local(link.strip())

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    repo = parse_link(link)
    api = HfApi(token=token)
    try:
        info = api.model_info(repo, files_metadata=True)
    except GatedRepoError:
        raise PatientRejected("gated", _gated_msg())
    except RepositoryNotFoundError:
        raise PatientRejected("not_found", f"Couldn't find a public model called '{repo}'. Check the link?")
    except RevisionNotFoundError:
        raise PatientRejected("not_found", f"Couldn't find that version of '{repo}'.")
    except HfHubHTTPError as e:
        raise PatientRejected("hub_error", f"Hugging Face didn't answer properly ({e.__class__.__name__}). Try again in a moment.")

    if getattr(info, "gated", False):
        raise PatientRejected("gated", _gated_msg())

    siblings = info.siblings or []
    # Only top-level weight files: subfolders usually hold duplicate copies
    # (e.g. "original/" checkpoints or ONNX exports) that transformers ignores.
    weights = [s for s in siblings if "/" not in s.rfilename and s.rfilename.endswith(_WEIGHT_SUFFIXES)]
    if not weights:
        has_bin = any(s.rfilename.endswith(".bin") for s in siblings)
        raise PatientRejected(
            "no_safetensors",
            "This model only ships older pickle (.bin) weights, which can run hidden code when loaded. "
            "The clinic only accepts safe .safetensors weights." if has_bin else
            "No model weights found in that repository (is it a GGUF/ONNX-only repo?).",
        )
    size = sum((s.size or 0) for s in weights)
    _check_size(size)

    try:
        cfg_file = hf_hub_download(repo, "config.json", revision=info.sha, token=token)
    except Exception:
        raise PatientRejected("no_config", "This repository has no config.json, so transformers can't load it.")
    with open(cfg_file) as f:
        cfg = json.load(f)
    arch, model_type, layers = _check_config(cfg)

    params, dtype = None, None
    st = getattr(info, "safetensors", None)
    if st is not None:
        params = getattr(st, "total", None)
        by_dtype = getattr(st, "parameters", None) or {}
        if by_dtype:
            dtype = max(by_dtype.items(), key=lambda kv: kv[1])[0]

    return Patient(
        repo_id=repo, revision=info.sha, link=f"https://huggingface.co/{repo}",
        size_bytes=size, params=params, dtype=dtype, architecture=arch,
        model_type=model_type, num_layers=layers,
    )


def _gated_msg() -> str:
    return (
        "This model is gated: its owners ask each person to accept their licence and use a personal access token. "
        "The shared clinic doesn't handle tokens. Clone the repo and run it with your own HF_TOKEN."
    )


def download_patterns() -> dict:
    """Files to fetch: weights, config and tokenizer, nothing from subfolders."""
    return {
        "allow_patterns": ["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken"],
        "ignore_patterns": ["*/*", "*.bin", "*.pth", "*.pt", "*.gguf", "*.onnx", "*.msgpack", "*.h5"],
    }
