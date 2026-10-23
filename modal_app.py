"""Deploy the Model Diet Clinic to Modal.

    modal deploy modal_app.py      # publish (prints your app's URL)
    modal serve modal_app.py       # live-reloading dev version

Three pieces run on Modal:

- web         a small CPU container serving the page and the API (always cheap)
- fetch_model a CPU container that downloads models into a shared cache
- treat       the GPU treatment room; it only runs (and only bills) during a diet

Limits live in diet/config.py and can be overridden with environment variables
in the `settings` secret-free dict below.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import modal

from diet import config

APP_NAME = "model-diet-clinic"
CACHE = "/cache"

# Settings forwarded to every container, so a forker can change limits with
# environment variables at deploy time, e.g.  DIET_MAX_MODEL_GB=30 modal deploy modal_app.py
settings = {k: v for k, v in os.environ.items() if k.startswith("DIET_")}
settings.update({
    "HF_HOME": f"{CACHE}/hf",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
})

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("diet-clinic-cache", create_if_missing=True)
cards = modal.Dict.from_name("diet-clinic-cards", create_if_missing=True)
limits = modal.Dict.from_name("diet-clinic-limits", create_if_missing=True)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]>=0.115", "huggingface_hub>=0.34", "hf_xet")
    .env(settings)
    .add_local_dir("diet", "/root/diet")
    .add_local_dir("docs", "/root/docs")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers>=4.56,<5",
        "accelerate>=1.0",
        "bitsandbytes>=0.46",
        "safetensors",
        "sentencepiece",
        "tiktoken",
        "protobuf",
        "huggingface_hub>=0.34",
        "hf_xet",
        "fastapi>=0.115",
    )
    .env(settings)
    .add_local_dir("diet", "/root/diet")
    .add_local_dir("docs", "/root/docs")
)


def _stream_from_thread(work):
    """Run `work(emit)` in a thread and yield whatever it emits."""
    q: queue.Queue = queue.Queue()
    done = object()

    def run():
        try:
            work(q.put)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            q.put(done)

    threading.Thread(target=run, daemon=True).start()
    last_beat = time.time()
    while True:
        try:
            e = q.get(timeout=5)
        except queue.Empty:
            # A small heartbeat keeps the stream alive during long silent steps.
            if time.time() - last_beat > 15:
                last_beat = time.time()
                yield {"type": "heartbeat"}
            continue
        if e is done:
            return
        yield e


@app.function(image=web_image, volumes={CACHE: volume}, cpu=2.0, memory=4096, timeout=60 * 60)
def fetch_model(repo_id: str, revision: str, size_bytes: int):
    from diet.fetch import fetch

    def work(emit):
        fetch(repo_id, revision, CACHE, size_bytes, emit)
        volume.commit()

    yield from _stream_from_thread(work)


@app.function(
    image=gpu_image,
    gpu=config.GPU,
    volumes={CACHE: volume},
    timeout=config.GPU_TIMEOUT_S,
    max_containers=config.MAX_GPU_CONTAINERS,
    scaledown_window=120,
)
def treat(job: dict):
    from diet.engine import treat as run_treatment
    from diet.fetch import model_dir

    def work(emit):
        volume.reload()
        path = model_dir(CACHE, job["repo_id"], job["revision"])
        res = run_treatment(path, job["patient"], job["diet"], job["plan"], job["options"], job["prompt"], emit)
        emit({"type": "result", "result": res})

    yield from _stream_from_thread(work)


@app.function(image=web_image, cpu=1.0, memory=1024, timeout=60 * 60, scaledown_window=15 * 60)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label=APP_NAME)
def web():
    from diet.api import MemoryLimiter, MemoryStore, create_app, summary

    class ModalStore(MemoryStore):
        """Weigh-in cards kept in a Modal Dict so every container sees them."""

        def __init__(self):
            self.folder = None

        async def save(self, result):
            await cards.put.aio(f"r:{result['id']}", result)
            recent = await cards.get.aio("recent", default=[]) or []
            await cards.put.aio("recent", ([summary(result)] + recent)[:24])

        async def get(self, rid):
            return await cards.get.aio(f"r:{rid}", default=None)

        async def recent(self):
            return await cards.get.aio("recent", default=[]) or []

    class ModalLimiter(MemoryLimiter):
        async def _get(self, key):
            return await limits.get.aio(key, default=None)

        async def _put(self, key, value):
            await limits.put.aio(key, value)

        async def _get_active(self, who):
            return await limits.get.aio(f"active:{who}", default=None)

        async def _set_active(self, who, value):
            await limits.put.aio(f"active:{who}", value or 0)

    async def runner(job):
        yield {"type": "progress", "stage": "check_in", "message": "Checking the patient in…"}
        async for e in fetch_model.remote_gen.aio(job["repo_id"], job["revision"], job["size_bytes"]):
            if e.get("type") == "heartbeat":
                continue
            yield e
            if e.get("type") == "error":
                return
        yield {"type": "progress", "stage": "lobby",
               "message": "Booking a treatment room (GPU). If the clinic is busy you'll wait in the lobby…"}
        async for e in treat.remote_gen.aio(job):
            if e.get("type") == "heartbeat":
                continue
            yield e

    return create_app(runner=runner, store=ModalStore(), limiter=ModalLimiter(), mode="modal")
