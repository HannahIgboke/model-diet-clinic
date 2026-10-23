import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from diet import hub  # noqa: E402
from diet.engine import treat  # noqa: E402
from tiny import make_model  # noqa: E402


@pytest.fixture(scope="session")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    return {
        "llama": make_model("llama", str(root / "llama"), chat=True),
        "llama32": make_model("llama", str(root / "llama32"), dtype=torch.float32),
        "qwen2": make_model("qwen2", str(root / "qwen2"), chat=True),
        "gpt2": make_model("gpt2", str(root / "gpt2")),
    }


def run(path, diet, plan, options=None, prompt=None):
    patient = hub.inspect_local(path).to_dict()
    events = []
    result = treat(path, patient, diet, plan, options or {}, prompt, emit=events.append)
    json.dumps(result)  # the card must be JSON-serialisable
    return result, events


# --- check-in -------------------------------------------------------------

@pytest.mark.parametrize("link,repo", [
    ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("https://hf.co/HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("https://huggingface.co/models/org/name/blob/main/config.json", "org/name"),
])
def test_parse_link(link, repo):
    assert hub.parse_link(link) == repo


@pytest.mark.parametrize("link,code", [
    ("", "empty"),
    ("https://github.com/org/repo", "unsupported_host"),
    ("https://huggingface.co/datasets/org/data", "not_a_model"),
    ("https://huggingface.co/Qwen", "bad_link"),
    ("not a link at all", "bad_link"),
])
def test_parse_link_rejects(link, code):
    with pytest.raises(hub.PatientRejected) as e:
        hub.parse_link(link)
    assert e.value.code == code


def test_size_cap(models, monkeypatch):
    monkeypatch.setattr(hub.config, "MAX_MODEL_GB", 1e-6)
    with pytest.raises(hub.PatientRejected) as e:
        hub.inspect_local(models["llama"])
    assert e.value.code == "too_big"


def test_custom_code_rejected():
    with pytest.raises(hub.PatientRejected) as e:
        hub._check_config({"architectures": ["FooForCausalLM"], "auto_map": {"AutoModel": "x.Foo"}})
    assert e.value.code == "custom_code"


def test_distill_cap(models, monkeypatch):
    monkeypatch.setattr(hub.config, "MAX_DISTILL_GB", 1e-6)
    p = hub.inspect_local(models["llama"]).to_dict()
    assert p["allowed_diets"]["distill"]["ok"] is False


# --- diets ------------------------------------------------------------------

def test_quantize_levels(models):
    r8, _ = run(models["llama"], "quantize", "int8")
    r4, _ = run(models["llama"], "quantize", "nf4")
    assert r8["after"]["bytes"] < r8["before"]["bytes"]
    assert r4["after"]["bytes"] < r8["after"]["bytes"]
    assert r4["after"]["params"] == r4["before"]["params"]          # every weight kept
    assert r8["health"]["agreement"] > r4["health"]["agreement"] - 0.2
    assert len(r4["samples"]) == 3


def test_half_portions_from_fp32(models):
    r, _ = run(models["llama32"], "quantize", "bf16")
    assert r["before"]["dtype"] == "float32"
    assert r["after"]["bytes"] == pytest.approx(r["before"]["bytes"] / 2, rel=0.01)


def test_magnitude_pruning_does_not_lose_weight(models):
    # Not offered in the clinic any more, but the function stays as a teaching example.
    from transformers import AutoModelForCausalLM
    from diet.measure import footprint_bytes, nonzero_fraction
    from diet.menu import BadPrescription, validate
    from diet.prune import magnitude_prune

    m = AutoModelForCausalLM.from_pretrained(models["llama"], torch_dtype=torch.bfloat16)
    pruned, x = magnitude_prune(m, 0.5)
    assert footprint_bytes(pruned) == footprint_bytes(m)            # the trap
    assert nonzero_fraction(pruned) < nonzero_fraction(m) - 0.2
    assert x["pruned_layer_bytes_if_sparse"] > x["pruned_layer_bytes_dense"]
    with pytest.raises(BadPrescription):
        validate("prune", "magnitude", {})


@pytest.mark.parametrize("kind", ["llama", "qwen2", "gpt2"])
def test_layer_pruning(models, kind):
    r, _ = run(models[kind], "prune", "layers", {"fraction": 0.34})
    x = r["extras"]
    assert x["layers_before"] == 6 and x["layers_after"] == 4
    assert 0 not in x["dropped_layers"] and 5 not in x["dropped_layers"]
    assert r["after"]["params"] < r["before"]["params"]
    assert r["after"]["bytes"] < r["before"]["bytes"]


@pytest.mark.parametrize("kind", ["qwen2", "gpt2"])
def test_quantize_other_architectures(models, kind):
    r, _ = run(models[kind], "quantize", "nf4", prompt="Tell me about rivers.")
    assert r["after"]["bytes"] < r["before"]["bytes"]
    assert r["samples"][0]["prompt"] == "Tell me about rivers."


def test_distillation(models):
    r, events = run(models["llama"], "distill", "student", {"keep_fraction": 0.5, "steps": 12})
    x = r["extras"]
    assert x["layers_after"] == 3
    assert len(x["loss_curve"]) == 12
    assert sum(e["type"] == "train" for e in events) == 12
    assert r["after"]["dtype"] == "bfloat16"                          # cast back after training
    assert r["after"]["bytes"] < r["before"]["bytes"]


# --- the API -----------------------------------------------------------------

def test_api_stream(models, tmp_path):
    from fastapi.testclient import TestClient
    from diet.local import build_app

    app = build_app(cache_root=str(tmp_path / "cache"), results_dir=str(tmp_path / "results"))
    c = TestClient(app)
    assert c.get("/").status_code == 200
    cfg = c.get("/api/config").json()
    assert cfg["max_model_gb"] == 15 and "quantize" in cfg["diets"]

    info = c.post("/api/inspect", json={"link": models["llama"]}).json()
    assert info["ok"] and info["patient"]["num_layers"] == 6
    bad = c.post("/api/inspect", json={"link": "https://github.com/x/y"})
    assert bad.status_code == 422 and bad.json()["code"] == "unsupported_host"

    events = []
    with c.stream("POST", "/api/diet", json={"link": models["llama"], "diet": "quantize", "plan": "nf4"}) as s:
        assert s.status_code == 200
        for line in s.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    types = [e["type"] for e in events]
    assert types[0] == "admitted" and types[-1] == "result", types
    rid = events[-1]["result"]["id"]
    assert c.get(f"/api/result/{rid}").json()["result"]["id"] == rid
    assert c.get("/api/recent").json()["recent"][0]["id"] == rid
    assert c.get(f"/r/{rid}", follow_redirects=False).headers["location"] == f"/?card={rid}"
    assert c.get("/img/hedge.webp").status_code == 200
    cfg = c.get("/api/config").json()
    assert cfg["diets"]["prune"]["name"] == "Trim the hedge" and list(cfg["diets"]["prune"]["plans"]) == ["layers"]
    assert os.path.isfile(tmp_path / "results" / f"{rid}.json")

    nope = c.post("/api/diet", json={"link": models["llama"], "diet": "juice-cleanse", "plan": "x"})
    assert nope.status_code == 422


def test_public_mode_blocks_local_paths(models, tmp_path):
    from fastapi.testclient import TestClient
    from diet.local import build_app

    c = TestClient(build_app(str(tmp_path / "c"), str(tmp_path / "r"), public=True))
    assert c.get("/api/config").json()["mode"] == "hosted"
    assert c.get("/api/config").json()["runs_per_day"] == 8
    r = c.post("/api/inspect", json={"link": models["llama"]})
    assert r.status_code == 422


# --- the gist noticeboard (Kaggle + GitHub Pages mode) ----------------------

@pytest.fixture()
def fake_github():
    import socket
    import threading
    import time

    import uvicorn
    from fake_github import make_app

    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    server = uvicorn.Server(uvicorn.Config(make_app(), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True); t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


def test_gist_noticeboard(models, tmp_path, fake_github):
    import time
    from fastapi.testclient import TestClient
    from diet.gist import Gist
    from diet.local import build_app, TreatmentRooms

    g = Gist(None, "test-token", api=fake_github)
    g.create()
    assert g.page_settings() == f"tester/{g.id}"
    g.announce("open", "https://example.trycloudflare.com")
    assert Gist(g.id, None, api=fake_github).read_file("clinic.json")["status"] == "open"

    with pytest.raises(RuntimeError, match="401"):
        Gist(g.id, "wrong", api=fake_github).announce("open")

    app = build_app(str(tmp_path / "c"), str(tmp_path / "r"), public=False, gist=Gist(g.id, "test-token", api=fake_github),
                    cors=["*"], rooms=TreatmentRooms(["cpu"]))
    c = TestClient(app)
    events = []
    with c.stream("POST", "/api/diet", json={"link": models["llama"], "diet": "prune", "plan": "layers"},
                  headers={"Origin": "https://someone.github.io"}) as s:
        assert s.headers.get("access-control-allow-origin") == "*"
        for line in s.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    rid = events[-1]["result"]["id"]
    for _ in range(50):                       # the gist copy is written in the background
        files = Gist(g.id, None, api=fake_github).fetch()["files"]
        if f"card-{rid}.json" in files:
            break
        time.sleep(0.1)
    reader = Gist(g.id, None, api=fake_github)
    assert reader.read_file(f"card-{rid}.json")["id"] == rid
    assert reader.read_file("recent.json")[0]["id"] == rid

    # A fresh server (a new Kaggle session) still finds the old card and the recent list.
    app2 = build_app(str(tmp_path / "c"), str(tmp_path / "r"), gist=Gist(g.id, "test-token", api=fake_github),
                     rooms=TreatmentRooms(["cpu"]))
    c2 = TestClient(app2)
    assert c2.get(f"/api/result/{rid}").json()["result"]["id"] == rid
    assert c2.get("/api/recent").json()["recent"][0]["id"] == rid
    assert c2.get("/config.js").status_code == 200


def test_rooms_queue():
    from diet.local import TreatmentRooms

    rooms = TreatmentRooms(["cuda:0", "cuda:1"])
    a, b = rooms.take(), rooms.take()
    assert {a, b} == {"cuda:0", "cuda:1"}
    waited = []
    import threading
    t = threading.Thread(target=lambda: waited.append(rooms.take(on_wait=lambda: waited.append("lobby"))))
    t.start()
    import time; time.sleep(0.2)
    assert waited == ["lobby"]
    rooms.give_back(a); t.join(2)
    assert waited == ["lobby", a]
