"""A tiny noticeboard on GitHub Gists, for when the clinic runs on borrowed GPUs.

The clinic page lives permanently on GitHub Pages, but the GPU (a Kaggle
notebook) comes and goes, and gets a new public address every time. A gist
holds the few things that must outlive a session:

    clinic.json        is the clinic open, and at which address?
    recent.json        the latest visits, for the "Recent visits" list
    card-<id>.json     every weigh-in card, so shared links keep working

The page only ever *reads* the gist (no token needed). Writing needs a GitHub
token with the "gist" permission, which stays in the notebook's secrets.
Only the standard library is used, so this works anywhere.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone

API = os.environ.get("DIET_GIST_API", "https://api.github.com")
MAX_CARDS = 200
RECENT = 24


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Gist:
    def __init__(self, gist_id: str | None, token: str | None, api: str | None = None):
        self.id = gist_id
        self.token = token
        self.api = (api or API).rstrip("/")
        self.owner = None
        self._lock = threading.Lock()

    # -- HTTP ---------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(self.api + path, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "model-diet-clinic")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, data=data, timeout=30) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            hint = ""
            if e.code in (401, 403):
                hint = " Check that the GitHub token is right and has the 'gist' permission."
            elif e.code == 404:
                hint = " Check the gist id (and that the token belongs to the gist's owner)."
            raise RuntimeError(f"GitHub said {e.code} for {method} {path}.{hint} {detail}") from None
        return json.loads(raw) if raw else {}

    # -- setup ----------------------------------------------------------------
    def create(self, description="Model Diet Clinic noticeboard") -> str:
        """Make a new (secret) gist and return its id."""
        g = self._request("POST", "/gists", {
            "description": description,
            "public": False,
            "files": {
                "clinic.json": {"content": json.dumps({"status": "closed", "updated_at": _now()}, indent=1)},
                "recent.json": {"content": "[]"},
            },
        })
        self.id = g["id"]
        self.owner = g["owner"]["login"]
        return self.id

    def fetch(self) -> dict:
        g = self._request("GET", f"/gists/{self.id}")
        self.owner = (g.get("owner") or {}).get("login")
        return g

    def file(self, gist: dict, name: str):
        f = (gist.get("files") or {}).get(name)
        if not f:
            return None
        content = f.get("content")
        if f.get("truncated") and f.get("raw_url"):
            with urllib.request.urlopen(f["raw_url"], timeout=30) as r:
                content = r.read().decode()
        try:
            return json.loads(content) if content else None
        except json.JSONDecodeError:
            return None

    def read_file(self, name: str):
        return self.file(self.fetch(), name)

    def write(self, files: dict):
        """files: {name: python object, or None to delete}."""
        payload = {
            name: (None if value is None else {"content": json.dumps(value, indent=None, default=float)})
            for name, value in files.items()
        }
        with self._lock:
            return self._request("PATCH", f"/gists/{self.id}", {"files": payload})

    # -- the noticeboard ------------------------------------------------------
    def announce(self, status: str, url: str | None = None, **extra):
        info = {"status": status, "url": url, "updated_at": _now(), **extra}
        self.write({"clinic.json": info})
        return info

    def page_settings(self) -> str:
        return f"{self.owner}/{self.id}" if self.owner else str(self.id)


class GistStore:
    """Weigh-in cards kept in memory and copied to the gist.

    Gist writes happen in the background, so a slow or failing GitHub never
    holds up a visit; failures are printed, not raised.
    """

    def __init__(self, gist: Gist):
        from .api import summary

        self._summary = summary
        self.g = gist
        self.items: dict[str, dict] = {}
        self.recent_list: list[dict] = []
        self.index: list[str] = []
        try:
            data = self.g.fetch()
            self.recent_list = self.g.file(data, "recent.json") or []
            self.index = sorted(
                [n[5:-5] for n in (data.get("files") or {}) if n.startswith("card-") and n.endswith(".json")],
                key=lambda rid: next((i for i, r in enumerate(self.recent_list) if r["id"] == rid), 10**6),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[gist] couldn't read the noticeboard yet: {e}")

    def _push(self, result: dict, recent: list[dict], drop: list[str]):
        try:
            files = {f"card-{result['id']}.json": result, "recent.json": recent}
            files.update({f"card-{rid}.json": None for rid in drop})
            self.g.write(files)
        except Exception as e:  # noqa: BLE001
            print(f"[gist] couldn't save card {result['id']}: {e}")

    async def save(self, result: dict):
        rid = result["id"]
        self.items[rid] = result
        self.recent_list = ([self._summary(result)] + [r for r in self.recent_list if r["id"] != rid])[:RECENT]
        self.index = [rid] + [i for i in self.index if i != rid]
        drop = self.index[MAX_CARDS:]
        self.index = self.index[:MAX_CARDS]
        threading.Thread(target=self._push, args=(result, list(self.recent_list), drop), daemon=True).start()

    async def get(self, rid: str):
        if rid in self.items:
            return self.items[rid]
        try:
            from fastapi.concurrency import run_in_threadpool

            r = await run_in_threadpool(self.g.read_file, f"card-{rid}.json")
        except Exception:  # noqa: BLE001
            r = None
        if r:
            self.items[rid] = r
        return r

    async def recent(self):
        return self.recent_list


def main():
    """Command line helper: create a gist, or announce the clinic's status."""
    import argparse

    ap = argparse.ArgumentParser(description="Model Diet Clinic gist noticeboard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create")
    a = sub.add_parser("announce")
    a.add_argument("status", choices=["open", "closed"])
    a.add_argument("--url")
    args = ap.parse_args()
    g = Gist(os.environ.get("DIET_GIST_ID"), os.environ.get("DIET_GIST_TOKEN"))
    if args.cmd == "create":
        g.create()
        print(g.page_settings())
    else:
        g.fetch()
        print(g.announce(args.status, args.url))


if __name__ == "__main__":
    main()
