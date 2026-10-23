"""Clinic settings.

Everything a forker is likely to want to change lives here, and every value can
be overridden with an environment variable, so you can raise the limits on your
own Modal account without touching the code.
"""

import os


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# --- Patient size limits ---------------------------------------------------
# Largest model (size of its weight files on the Hub) the clinic will accept.
MAX_MODEL_GB = _num("DIET_MAX_MODEL_GB", 15)

# Distillation trains a student model, which needs far more memory than the
# other diets (gradients + optimizer state), so it has its own, smaller cap.
MAX_DISTILL_GB = _num("DIET_MAX_DISTILL_GB", 3)

# Models at or above this size count as "big" for the daily limits below.
BIG_MODEL_GB = _num("DIET_BIG_MODEL_GB", 4)

# --- Fair-use limits (per visitor, per day) --------------------------------
# Set DIET_RATE_LIMIT=0 to switch them off (handy when running locally).
RATE_LIMIT_ENABLED = os.environ.get("DIET_RATE_LIMIT", "1") != "0"
RUNS_PER_DAY = int(_num("DIET_RUNS_PER_DAY", 8))
BIG_RUNS_PER_DAY = int(_num("DIET_BIG_RUNS_PER_DAY", 2))
# Only one visit at a time per visitor. Turn off (0) when many people share one
# network, like a meetup on venue wifi: they all look like the same visitor.
ONE_VISIT_AT_A_TIME = os.environ.get("DIET_ONE_VISIT_AT_A_TIME", "1") != "0"

# --- Hardware ----------------------------------------------------------------
# GPU type for the treatment room. L40S (48GB) holds a 15GB model and its
# dieted copy side by side. Use "A100-80GB" for more headroom.
GPU = os.environ.get("DIET_GPU", "L40S")

# How many GPUs may run at once. Extra visitors wait in the lobby (queue)
# instead of spinning up more GPUs and burning credits.
MAX_GPU_CONTAINERS = int(_num("DIET_MAX_GPU_CONTAINERS", 3))

# Longest a single treatment may take before it is stopped (seconds).
GPU_TIMEOUT_S = int(_num("DIET_GPU_TIMEOUT_S", 20 * 60))

# Downloaded models are cached on a Modal Volume. When the cache grows past
# this size, the least recently used models are evicted.
CACHE_LIMIT_GB = _num("DIET_CACHE_LIMIT_GB", 150)

# --- Evaluation ----------------------------------------------------------------
EVAL_MAX_TOKENS = int(_num("DIET_EVAL_MAX_TOKENS", 1024))
EVAL_CHUNK = int(_num("DIET_EVAL_CHUNK", 256))
SAMPLE_NEW_TOKENS = int(_num("DIET_SAMPLE_NEW_TOKENS", 60))
