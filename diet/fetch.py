"""Bring the patient in: download a model into the cache, once.

Downloads run on a CPU machine (cheap) into a shared cache, so when twenty
people at a meetup try the same model it is fetched only the first time.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid

from . import config
from .hub import download_patterns

MARKER = ".clinic-complete"


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def model_dir(cache_root: str, repo_id: str, revision: str) -> str:
    return os.path.join(cache_root, "models", repo_id.replace("/", "__"), revision)


def evict(cache_root: str, keep: str | None = None, limit_gb: float | None = None):
    """Delete least recently used models until the cache fits under the limit."""
    limit = (limit_gb if limit_gb is not None else config.CACHE_LIMIT_GB) * 1024**3
    base = os.path.join(cache_root, "models")
    if not os.path.isdir(base):
        return []
    entries = []
    for repo in os.listdir(base):
        for rev in os.listdir(os.path.join(base, repo)):
            d = os.path.join(base, repo, rev)
            m = os.path.join(d, MARKER)
            if os.path.isfile(m):
                entries.append((os.path.getmtime(m), d, _dir_size(d)))
    total = sum(e[2] for e in entries)
    removed = []
    for _, d, size in sorted(entries):
        if total <= limit:
            break
        if keep and os.path.abspath(d) == os.path.abspath(keep):
            continue
        shutil.rmtree(d, ignore_errors=True)
        total -= size
        removed.append(d)
    return removed


def fetch(repo_id: str, revision: str, cache_root: str, expected_bytes: int = 0, emit=lambda e: None) -> str:
    target = model_dir(cache_root, repo_id, revision)
    marker = os.path.join(target, MARKER)
    if os.path.isfile(marker):
        os.utime(marker)  # mark as recently used
        emit({"type": "progress", "stage": "check_in", "message": "Patient already in the waiting room (cached). No download needed.", "pct": 1.0, "download": {"done": expected_bytes, "total": expected_bytes, "cached": True}})
        return target

    from huggingface_hub import snapshot_download

    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = f"{target}.partial-{uuid.uuid4().hex[:6]}"
    stop = threading.Event()

    def report():
        t0 = time.time()
        while not stop.wait(1.0):
            done = _dir_size(tmp)
            elapsed = max(time.time() - t0, 1e-6)
            emit({
                "type": "progress", "stage": "check_in",
                "message": "Bringing the patient in (downloading)…",
                "download": {"done": done, "total": expected_bytes, "rate": done / elapsed},
                "pct": min(done / expected_bytes, 0.99) if expected_bytes else None,
            })

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    try:
        snapshot_download(repo_id, revision=revision, local_dir=tmp, **download_patterns())
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        stop.set()
        reporter.join(timeout=2)

    shutil.rmtree(os.path.join(tmp, ".cache"), ignore_errors=True)
    if os.path.isdir(target):          # someone else finished first
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        os.rename(tmp, target)
    with open(marker, "w") as f:
        f.write(str(time.time()))
    emit({"type": "progress", "stage": "check_in", "message": "Patient has arrived.", "pct": 1.0,
          "download": {"done": expected_bytes, "total": expected_bytes}})
    evict(cache_root, keep=target)
    return target
