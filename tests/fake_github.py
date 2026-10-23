"""A stand-in for GitHub's gist API, so the noticeboard can be tested offline.

    python tests/fake_github.py 9100     # serves the API and raw files on that port
"""

from __future__ import annotations

import json
import sys
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

TOKEN = "test-token"
OWNER = "tester"


def make_app():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    gists: dict[str, dict] = {}

    def authed(req: Request):
        return req.headers.get("authorization") == f"Bearer {TOKEN}"

    def view(gid):
        g = gists[gid]
        return {"id": gid, "owner": {"login": OWNER},
                "files": {n: {"filename": n, "content": c, "truncated": False,
                              "raw_url": f"/{OWNER}/{gid}/raw/{n}"} for n, c in g.items()}}

    @app.post("/gists")
    async def create(req: Request):
        if not authed(req):
            return JSONResponse({"message": "Bad credentials"}, 401)
        body = await req.json()
        gid = uuid.uuid4().hex
        gists[gid] = {n: f["content"] for n, f in body["files"].items()}
        return view(gid)

    @app.get("/gists/{gid}")
    async def get(gid: str):
        if gid not in gists:
            return JSONResponse({"message": "Not Found"}, 404)
        return view(gid)

    @app.patch("/gists/{gid}")
    async def patch(gid: str, req: Request):
        if not authed(req):
            return JSONResponse({"message": "Bad credentials"}, 401)
        if gid not in gists:
            return JSONResponse({"message": "Not Found"}, 404)
        for n, f in (await req.json())["files"].items():
            if f is None:
                gists[gid].pop(n, None)
            else:
                gists[gid][n] = f["content"]
        return view(gid)

    @app.get("/{owner}/{gid}/raw/{name}")
    async def raw(owner: str, gid: str, name: str):
        if gid not in gists or name not in gists[gid]:
            return PlainTextResponse("404: Not Found", 404)
        return PlainTextResponse(gists[gid][name])

    app.state.gists = gists
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(make_app(), host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 9100, log_level="warning")
