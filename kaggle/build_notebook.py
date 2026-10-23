"""Builds kaggle/model_diet_clinic_kaggle.ipynb from the cells below.

Edit the cells here, then run:  python kaggle/build_notebook.py
"""

import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# Model Diet Clinic: treatment room on Kaggle

This notebook opens the clinic: it runs the dieting engine on Kaggle's free GPUs and connects it to
your clinic page on GitHub Pages. While it runs, anyone with the page's link can put a model on a diet.
When it stops, the page says "Closed", and weigh-in cards from earlier visits still open.

**Before the first run** (in the panel on the right, under *Session options*):

1. **Accelerator:** choose **GPU T4 x2**.
2. **Internet:** switch it **on** (this needs a phone-verified Kaggle account).
3. **Secrets** (Add-ons menu → Secrets): add a secret called `GIST_TOKEN` holding your GitHub token,
   and tick the box to attach it to this notebook.

Then fill in the settings in the next cell and run everything (**Run All**).

**To keep the clinic open without your laptop** (for example during a talk), use
**Save Version → Save & Run All (Commit)**. It runs in the background for up to `HOURS_OPEN` hours.
""")

code(r"""
# ---- Settings -----------------------------------------------------------------------
GITHUB_REPO = "YOUR-USERNAME/model-diet-clinic"   # your copy of the code on GitHub

# Leave GIST empty on the very first run. The notebook creates one and prints a value
# like "your-username/0123abcd...". Paste it here AND into docs/config.js on GitHub.
GIST = ""

HOURS_OPEN = 6          # the clinic closes itself after this long (Kaggle's limit is 12)

# Kaggle's T4 GPUs have 16GB each, so these are the safe ceilings.
MAX_MODEL_GB = 6        # biggest model accepted (Qwen2.5-3B fits; 7B models don't)
MAX_DISTILL_GB = 1.5    # distillation trains a model, which needs far more memory
BIG_MODEL_GB = 4        # models above this count as "big" for the daily limits

# At a meetup everyone shares the venue's wifi, so they all look like ONE visitor.
# These are generous on purpose. Lower them for an always-on public clinic.
RUNS_PER_DAY = 60
BIG_RUNS_PER_DAY = 20
""")

md("## 1. Check the machine")
code(r"""
import os, sys, subprocess, time, json, re, threading, urllib.request

import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU found. In the right panel, set Accelerator to 'GPU T4 x2', then run again.")
GPUS = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPUs:", GPUS)
if any("P100" in g for g in GPUS):
    print("Heads up: the P100 doesn't support 4-bit quantization well. Choose 'GPU T4 x2' instead.")

try:
    urllib.request.urlopen("https://huggingface.co", timeout=15)
    print("Internet: on")
except Exception:
    raise SystemExit("No internet. In the right panel, switch Internet on (needs a phone-verified account).")

try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GIST_TOKEN")
except Exception as e:
    TOKEN = os.environ.get("GIST_TOKEN")
if not TOKEN:
    raise SystemExit("No GIST_TOKEN secret. Add it under Add-ons → Secrets and attach it to this notebook.")
print("GitHub token: found")
""")

md("## 2. Install and download the clinic")
code(r"""
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.56,<5", "accelerate>=1.0", "bitsandbytes>=0.46",
                "huggingface_hub>=0.34", "hf_xet", "safetensors", "sentencepiece",
                "fastapi>=0.115", "uvicorn[standard]"], check=True)

CLINIC_DIR = "/kaggle/working/clinic"
if os.path.isdir(os.path.join(CLINIC_DIR, ".git")):
    subprocess.run(["git", "-C", CLINIC_DIR, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{GITHUB_REPO}.git", CLINIC_DIR], check=True)
sys.path.insert(0, CLINIC_DIR)
print("Clinic code ready in", CLINIC_DIR)
""")

md("## 3. The noticeboard (a GitHub gist)")
code(r"""
from diet.gist import Gist

board = Gist(GIST.split("/")[-1].strip() or None, TOKEN)
if not GIST.strip():
    board.create()
    GIST = board.page_settings()
    print("=" * 72)
    print("Created your noticeboard. Do these two things once:")
    print(f'  1. In the settings cell above, set   GIST = "{GIST}"')
    print(f'  2. On GitHub, edit docs/config.js so it says   gist: "{GIST}",')
    print("=" * 72)
else:
    board.fetch()
    GIST = board.page_settings()
    print("Using noticeboard", GIST)
PAGE_URL = f"https://{GITHUB_REPO.split('/')[0].lower()}.github.io/{GITHUB_REPO.split('/')[1]}/"
""")

md("## 4. Open the treatment rooms")
code(r"""
env = dict(os.environ,
    DIET_GIST_ID=board.id, DIET_GIST_TOKEN=TOKEN,
    DIET_MAX_MODEL_GB=str(MAX_MODEL_GB), DIET_MAX_DISTILL_GB=str(MAX_DISTILL_GB),
    DIET_BIG_MODEL_GB=str(BIG_MODEL_GB), DIET_RUNS_PER_DAY=str(RUNS_PER_DAY),
    DIET_BIG_RUNS_PER_DAY=str(BIG_RUNS_PER_DAY), DIET_ONE_VISIT_AT_A_TIME="0",
    DIET_CACHE_LIMIT_GB="40", DIET_SALT=board.id,
    HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")

server_log = open("/kaggle/working/clinic-server.log", "a")
server = subprocess.Popen(
    [sys.executable, "-m", "diet.local", "--public", "--gist", "--cors", "*", "--port", "8000",
     "--cache", "/tmp/clinic-cache"],
    cwd=CLINIC_DIR, env=env, stdout=server_log, stderr=subprocess.STDOUT)

def tail(path, n=30):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""

for _ in range(120):
    if server.poll() is not None:
        print(tail("/kaggle/working/clinic-server.log"))
        raise SystemExit("The clinic server stopped while starting. The log is above.")
    try:
        cfg = json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/config", timeout=3))
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("The clinic server didn't start within 2 minutes.")
print(f"Clinic server running with {cfg['rooms']} treatment room(s), max model {cfg['max_model_gb']} GB.")
""")

md("## 5. Open the front door (public tunnel)")
code(r"""
TUNNEL_LOG = "/kaggle/working/clinic-tunnel.log"

def _wait_for_url(proc, pattern, timeout=90):
    found = {}
    def reader():
        with open(TUNNEL_LOG, "a") as log:
            for line in proc.stdout:
                log.write(line); log.flush()
                m = re.search(pattern, line)
                if m and "url" not in found:
                    found["url"] = m.group(0)
    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    while time.time() - t0 < timeout and "url" not in found and proc.poll() is None:
        time.sleep(0.5)
    return found.get("url")

def start_cloudflare():
    exe = "/tmp/cloudflared"
    if not os.path.exists(exe):
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", exe)
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--protocol", "http2",
                             "--url", "http://127.0.0.1:8000"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, _wait_for_url(proc, r"https://[-a-z0-9]+\.trycloudflare\.com")

def start_localhost_run():
    # Backup tunnel over SSH, in case Cloudflare's can't connect from Kaggle.
    proc = subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
                             "-R", "80:127.0.0.1:8000", "nokey@localhost.run"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, _wait_for_url(proc, r"https://[-a-z0-9]+\.lhr\.life")

def reachable(url, tries=12):
    for _ in range(tries):
        try:
            json.load(urllib.request.urlopen(url + "/api/config", timeout=10))
            return True
        except Exception:
            time.sleep(5)
    return False

def open_tunnel():
    for name, start in (("Cloudflare", start_cloudflare), ("localhost.run", start_localhost_run)):
        try:
            proc, url = start()
        except Exception as e:
            print(f"{name} tunnel failed to start: {e}")
            continue
        if url and reachable(url):
            print(f"{name} tunnel open: {url}")
            return proc, url
        print(f"{name} tunnel didn't come up. Its log:\n{tail(TUNNEL_LOG, 15)}")
        proc.kill()
    raise SystemExit("Couldn't open a public tunnel from Kaggle. Send the log above to whoever is helping you.")

tunnel, PUBLIC_URL = open_tunnel()
""")

md("## 6. Announce: the clinic is open")
code(r"""
from datetime import datetime, timezone, timedelta

closes = datetime.now(timezone.utc) + timedelta(hours=HOURS_OPEN)
board.announce("open", PUBLIC_URL, gpus=GPUS, closes_at=closes.isoformat(timespec="minutes"))
print("=" * 72)
print("The clinic is OPEN.")
print("  Your clinic page:   ", PAGE_URL)
print("  Direct link (today):", PUBLIC_URL)
print(f"  Closes by itself at {closes.strftime('%H:%M')} UTC. Stop this cell to close it sooner.")
print("=" * 72)
""")

md(r"""
## 7. Keep the clinic open

This cell keeps everything running and checks on it every minute. It restarts the tunnel if it drops.
**Stop this cell** (or the session) to close the clinic; the page then shows "Closed".
""")
code(r"""
end = time.time() + HOURS_OPEN * 3600
failures, last_note = 0, 0
try:
    while time.time() < end:
        time.sleep(60)
        if server.poll() is not None:
            print(tail("/kaggle/working/clinic-server.log"))
            print("The clinic server stopped. The log is above.")
            break
        if reachable(PUBLIC_URL, tries=2):
            failures = 0
        else:
            failures += 1
            if failures >= 3:
                print("The tunnel stopped answering. Opening a new one…")
                tunnel.kill()
                tunnel, PUBLIC_URL = open_tunnel()
                board.announce("open", PUBLIC_URL, gpus=GPUS, closes_at=closes.isoformat(timespec="minutes"))
                failures = 0
        if time.time() - last_note > 600:
            last_note = time.time()
            left = (end - time.time()) / 3600
            visits = len(re.findall(r"POST /api/diet", tail("/kaggle/working/clinic-server.log", 100000)))
            print(f"{time.strftime('%H:%M', time.gmtime())} UTC  open at {PUBLIC_URL}  visits so far: {visits}  hours left: {left:.1f}")
except KeyboardInterrupt:
    print("Stopping on request.")
finally:
    try:
        board.announce("closed")
        print("Noticeboard updated: the clinic is CLOSED.")
    except Exception as e:
        print("Couldn't update the noticeboard:", e)
    for p in (tunnel, server):
        try:
            p.terminate()
        except Exception:
            pass
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_diet_clinic_kaggle.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
