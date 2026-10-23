"""Run the clinic on any machine: python -m diet.local

Uses your own CPU or GPU(s), keeps weigh-in cards in ./results, caches models
in ./cache, and switches the fair-use limits off. You can also paste a path to
a local model folder instead of a Hugging Face link.

With --public it's ready to share (limits on, no local folder paths), and with
--gist it keeps weigh-in cards on a GitHub gist so they outlive the machine;
that's how the Kaggle notebook runs it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import queue
import threading
import traceback


class TreatmentRooms:
    """One treatment room per GPU (or a single room on CPU).

    Each visit borrows a free room; when all are busy, visitors wait in the lobby.
    Set DIET_WORKERS=1 to use just one GPU even when there are more.
    """

    def __init__(self, devices: list[str] | None = None):
        if devices is None:
            import torch

            n = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(n)] or ["cpu"]
            limit = int(os.environ.get("DIET_WORKERS", "0") or 0)
            if limit > 0:
                devices = devices[:limit]
        self.devices = devices
        self.free: queue.Queue = queue.Queue()
        for d in devices:
            self.free.put(d)

    def take(self, on_wait=None) -> str:
        try:
            return self.free.get_nowait()
        except queue.Empty:
            if on_wait:
                on_wait()
            return self.free.get()

    def give_back(self, device: str):
        self.free.put(device)


def make_runner(cache_root: str, rooms: TreatmentRooms | None = None):
    from .engine import treat
    from .fetch import fetch

    rooms = rooms or TreatmentRooms()

    async def runner(job):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def emit(e):
            loop.call_soon_threadsafe(q.put_nowait, e)

        def work():
            device = None
            try:
                # Download first: it doesn't need a GPU, so it doesn't hold one up.
                if job["is_local"]:
                    path = job["repo_id"]
                else:
                    path = fetch(job["repo_id"], job["revision"], cache_root, job["size_bytes"], emit)
                device = rooms.take(on_wait=lambda: emit({
                    "type": "progress", "stage": "lobby",
                    "message": "Every treatment room is busy. Waiting in the lobby for the next free GPU…",
                }))
                res = treat(path, job["patient"], job["diet"], job["plan"], job["options"], job["prompt"],
                            emit, device=device)
                emit({"type": "result", "result": res})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                if device is not None:
                    rooms.give_back(device)
                emit(None)

        loop.run_in_executor(None, work)
        while True:
            e = await q.get()
            if e is None:
                break
            yield e

    return runner


def build_app(cache_root="cache", results_dir="results", limits=False, public=False,
              gist=None, cors=None, rooms=None):
    """public=True is for sharing the clinic: fair-use limits are on and visitors
    can't point the clinic at folders on the machine. gist is a diet.gist.Gist."""
    from .api import MemoryLimiter, MemoryStore, create_app

    if gist is not None:
        from .gist import GistStore

        store = GistStore(gist)
    else:
        store = MemoryStore(results_dir)
    rooms = rooms or TreatmentRooms()
    return create_app(
        runner=make_runner(os.path.abspath(cache_root), rooms),
        store=store,
        limiter=MemoryLimiter(enabled=limits or public),
        allow_local=not public,
        mode="hosted" if public else "local",
        cors_origins=cors,
        extra_config={"rooms": len(rooms.devices)},
    )


def main():
    ap = argparse.ArgumentParser(description="Run the Model Diet Clinic.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cache", default="cache", help="where downloaded models are kept")
    ap.add_argument("--results", default="results", help="where weigh-in cards are saved")
    ap.add_argument("--limits", action="store_true", help="enforce the fair-use limits locally too")
    ap.add_argument("--public", action="store_true",
                    help="share with the world: listen on all interfaces, limits on, no local folder paths")
    ap.add_argument("--gist", action="store_true",
                    help="keep weigh-in cards on the gist in DIET_GIST_ID (token in DIET_GIST_TOKEN)")
    ap.add_argument("--cors", default=os.environ.get("DIET_CORS_ORIGINS", ""),
                    help="comma-separated origins allowed to call the API, e.g. https://you.github.io (or *)")
    args = ap.parse_args()

    gist = None
    if args.gist:
        from .gist import Gist

        gist = Gist(os.environ.get("DIET_GIST_ID"), os.environ.get("DIET_GIST_TOKEN"))
    cors = [o.strip() for o in args.cors.split(",") if o.strip()] or None

    import uvicorn

    host = "0.0.0.0" if args.public and args.host == "127.0.0.1" else args.host
    app = build_app(args.cache, args.results, args.limits, args.public, gist, cors)
    uvicorn.run(app, host=host, port=args.port, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
