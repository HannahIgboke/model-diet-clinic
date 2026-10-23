"""The front desk: a FastAPI app that serves the clinic page and its API.

The same app runs locally (python -m diet.local) and on Modal (modal_app.py).
Only three things differ between the two, and they are passed in:

- runner:  async generator that takes a job and yields progress events
- store:   where finished weigh-in cards are kept (and the "recent" list)
- limiter: fair-use limits per visitor
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .hub import PatientRejected, inspect_model
from .menu import DIETS, BadPrescription, validate

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


class InspectBody(BaseModel):
    link: str = Field(max_length=500)


class DietBody(BaseModel):
    link: str = Field(max_length=500)
    diet: str
    plan: str
    options: dict = Field(default_factory=dict)
    prompt: str | None = Field(default=None, max_length=500)


class Limited(Exception):
    pass


# ---------------------------------------------------------------------------
# Simple in-memory store and limiter (used locally; Modal versions subclass)
# ---------------------------------------------------------------------------

def summary(result: dict) -> dict:
    return {
        "id": result["id"],
        "repo_id": result["patient"]["repo_id"],
        "diet": result["diet"]["name"],
        "plan": result["diet"]["label"],
        "before_gb": result["before"]["gb"],
        "after_gb": result["after"]["gb"],
        "percent_lost": result["change"]["percent_lost"],
        "verdict": result["health"]["level"],
        "created_at": result["created_at"],
    }


class MemoryStore:
    def __init__(self, folder: str | None = None):
        self.items: dict[str, dict] = {}
        self.recent_list: list[dict] = []
        self.folder = folder
        if folder:
            os.makedirs(folder, exist_ok=True)
            for f in sorted(os.listdir(folder), key=lambda n: os.path.getmtime(os.path.join(folder, n))):
                if f.endswith(".json"):
                    with open(os.path.join(folder, f)) as fh:
                        r = json.load(fh)
                    self.items[r["id"]] = r
                    self.recent_list.insert(0, summary(r))

    async def save(self, result: dict):
        self.items[result["id"]] = result
        self.recent_list = ([summary(result)] + self.recent_list)[:24]
        if self.folder:
            with open(os.path.join(self.folder, f"{result['id']}.json"), "w") as fh:
                json.dump(result, fh)

    async def get(self, rid: str) -> dict | None:
        return self.items.get(rid)

    async def recent(self) -> list[dict]:
        return self.recent_list


class MemoryLimiter:
    """Per-visitor daily limits plus one treatment at a time per visitor."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = config.RATE_LIMIT_ENABLED if enabled is None else enabled
        self.counts: dict[str, dict] = {}
        self.active: dict[str, float] = {}

    async def _get(self, key):
        return self.counts.get(key)

    async def _put(self, key, value):
        self.counts[key] = value

    async def _get_active(self, who):
        return self.active.get(who)

    async def _set_active(self, who, value):
        if value is None:
            self.active.pop(who, None)
        else:
            self.active[who] = value

    async def admit(self, who: str, size_gb: float):
        if not self.enabled:
            return
        started = await self._get_active(who) if config.ONE_VISIT_AT_A_TIME else None
        if started and time.time() - started < config.GPU_TIMEOUT_S + 600:
            raise Limited("You already have a patient in treatment. Wait for that visit to finish first.")
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"day:{day}:{who}"
        c = await self._get(key) or {"runs": 0, "big": 0}
        big = size_gb >= config.BIG_MODEL_GB
        if c["runs"] >= config.RUNS_PER_DAY:
            raise Limited(f"You've used all {config.RUNS_PER_DAY} of today's visits. Come back tomorrow, or clone the repo to run your own clinic.")
        if big and c["big"] >= config.BIG_RUNS_PER_DAY:
            raise Limited(
                f"Big patients (over {config.BIG_MODEL_GB:g}GB) are limited to {config.BIG_RUNS_PER_DAY} a day per visitor. "
                "Try a smaller model, or clone the repo to run your own clinic."
            )
        c["runs"] += 1
        c["big"] += int(big)
        await self._put(key, c)
        await self._set_active(who, time.time())

    async def release(self, who: str):
        if self.enabled:
            await self._set_active(who, None)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

def _visitor(request: Request) -> str:
    # Behind a Cloudflare tunnel the real visitor is in CF-Connecting-IP.
    ip = request.headers.get("cf-connecting-ip", "").strip()
    if not ip:
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")
    # Store a hash, never the raw IP address.
    salt = os.environ.get("DIET_SALT", "model-diet-clinic")
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:20]


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=float)}\n\n"


def create_app(runner, store=None, limiter=None, allow_local: bool = False, mode: str = "local",
               cors_origins: list[str] | None = None, extra_config: dict | None = None) -> FastAPI:
    store = store or MemoryStore()
    limiter = limiter or MemoryLimiter()
    app = FastAPI(title="Model Diet Clinic", docs_url="/api/docs", redoc_url=None)

    if cors_origins:
        # Lets the page on GitHub Pages talk to this server on another address.
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type"], max_age=600)

    app.mount("/img", StaticFiles(directory=os.path.join(STATIC, "img")), name="img")

    @app.get("/config.js", include_in_schema=False)
    async def page_config():
        # Served by this server, the page talks to this server: no gist needed.
        return FileResponse(os.path.join(STATIC, "config.js"), media_type="text/javascript",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-cache"})

    @app.get("/r/{rid}", include_in_schema=False)
    async def shared_card(rid: str):
        # Old-style card links: send them to the page with ?card=, so relative picture paths work.
        return RedirectResponse(f"/?card={rid}")

    @app.get("/api/config")
    async def get_config():
        return {
            "mode": mode,
            "max_model_gb": config.MAX_MODEL_GB,
            "max_distill_gb": config.MAX_DISTILL_GB,
            "big_model_gb": config.BIG_MODEL_GB,
            "runs_per_day": config.RUNS_PER_DAY if limiter.enabled else None,
            "big_runs_per_day": config.BIG_RUNS_PER_DAY if limiter.enabled else None,
            "gpu": config.GPU if mode == "modal" else ("this server" if mode == "hosted" else "your machine"),
            "diets": DIETS,
            **(extra_config or {}),
        }

    @app.post("/api/inspect")
    async def inspect(body: InspectBody):
        try:
            p = await run_in_threadpool(inspect_model, body.link, allow_local)
        except PatientRejected as e:
            return JSONResponse({"ok": False, "code": e.code, "message": e.message}, status_code=422)
        return {"ok": True, "patient": p.to_dict()}

    @app.get("/api/result/{rid}")
    async def result(rid: str):
        r = await store.get(rid)
        if not r:
            return JSONResponse({"ok": False, "message": "No weigh-in card with that id."}, status_code=404)
        return {"ok": True, "result": r}

    @app.get("/api/recent")
    async def recent():
        return {"ok": True, "recent": await store.recent()}

    @app.post("/api/diet")
    async def start_diet(body: DietBody, request: Request):
        try:
            opts = validate(body.diet, body.plan, body.options)
            patient = await run_in_threadpool(inspect_model, body.link, allow_local)
        except BadPrescription as e:
            return JSONResponse({"ok": False, "code": "bad_prescription", "message": str(e)}, status_code=422)
        except PatientRejected as e:
            return JSONResponse({"ok": False, "code": e.code, "message": e.message}, status_code=422)

        allowed = patient.to_dict()["allowed_diets"][body.diet]
        if not allowed.get("ok"):
            return JSONResponse({"ok": False, "code": "diet_not_allowed", "message": allowed.get("reason")}, status_code=422)

        who = _visitor(request)
        try:
            await limiter.admit(who, patient.size_gb)
        except Limited as e:
            return JSONResponse({"ok": False, "code": "limited", "message": str(e)}, status_code=429)

        job = {
            "patient": patient.to_dict(),
            "repo_id": patient.repo_id,
            "revision": patient.revision,
            "is_local": patient.is_local,
            "size_bytes": patient.size_bytes,
            "diet": body.diet,
            "plan": body.plan,
            "options": opts,
            "prompt": body.prompt,
        }

        async def stream():
            agen = runner(job).__aiter__()
            pending = None
            try:
                yield _sse({"type": "admitted", "patient": job["patient"]})
                while True:
                    if pending is None:
                        pending = asyncio.ensure_future(agen.__anext__())
                    done, _ = await asyncio.wait({pending}, timeout=10)
                    if not done:
                        yield ": still here\n\n"   # keep-alive for proxies
                        continue
                    try:
                        event = pending.result()
                    except StopAsyncIteration:
                        pending = None
                        break
                    except Exception as e:  # noqa: BLE001
                        pending = None
                        yield _sse({"type": "error", "message": f"The treatment failed: {e}"})
                        break
                    pending = None
                    if event.get("type") == "result":
                        await store.save(event["result"])
                    yield _sse(event)
            finally:
                if pending is not None:
                    pending.cancel()
                await limiter.release(who)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
