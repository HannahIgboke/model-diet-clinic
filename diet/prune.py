"""Diet 2: Pruning, or "cutting things out".

Two very different versions, on purpose:

- magnitude: zero out the smallest weights. The model gets *sparser*, but the
  scale doesn't move, because a zero still takes up as much space as any other
  number in a dense tensor. This is the crash diet that doesn't work.
- layers: remove whole transformer blocks. The model genuinely loses weight.
  Which blocks? The ones that change their input the least, measured with the
  "Block Influence" score from ShortGPT (Men et al., 2024).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as tprune

from .measure import device_of
from .quantize import _conv1d_cls, _skip

PLANS = {
    "magnitude": {"label": "Zero out the smallest weights (not offered in the clinic)"},
    "layers": {"label": "Cut the dead layers"},
}

_LAYER_PATHS = (
    "model.layers",
    "model.decoder.layers",
    "model.language_model.layers",
    "transformer.h",
    "transformer.layers",
    "transformer.blocks",
    "gpt_neox.layers",
)


def find_layers(model) -> tuple[nn.Module, str, nn.ModuleList]:
    """Find the stack of transformer blocks, whatever the architecture calls it."""
    for path in _LAYER_PATHS:
        parent_path, _, attr = path.rpartition(".")
        try:
            parent = model.get_submodule(parent_path)
        except AttributeError:
            continue
        layers = getattr(parent, attr, None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return parent, attr, layers
    # Fallback: the longest ModuleList in the model.
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and (best is None or len(mod) > len(best[2])):
            parent_name, _, attr = name.rpartition(".")
            best = (model.get_submodule(parent_name) if parent_name else model, attr, mod)
    if best is None:
        raise ValueError("Couldn't find the model's transformer layers.")
    return best


def magnitude_prune(original, amount: float = 0.5):
    dieted = copy.deepcopy(original)
    Conv1D = _conv1d_cls()
    dense_bytes = 0
    for name, mod in dieted.named_modules():
        if _skip(name, mod, dieted):
            continue
        if isinstance(mod, nn.Linear) or (Conv1D and isinstance(mod, Conv1D)):
            tprune.l1_unstructured(mod, name="weight", amount=amount)
            tprune.remove(mod, "weight")
            dense_bytes += mod.weight.numel() * mod.weight.element_size()
    # What if we stored only the non-zeros? Each needs its value plus a 4-byte
    # column index (CSR-style), so at 50% sparsity it's *heavier* than dense.
    elem = next(dieted.parameters()).element_size()
    kept = dense_bytes / elem * (1 - amount)
    sparse_bytes = int(kept * (elem + 4))
    return dieted, {
        "plan": "magnitude",
        "label": PLANS["magnitude"]["label"],
        "amount": amount,
        "pruned_layer_bytes_dense": int(dense_bytes),
        "pruned_layer_bytes_if_sparse": sparse_bytes,
    }


@torch.no_grad()
def block_influence(model, chunks) -> list[float]:
    """For each block: 1 - cosine similarity between what goes in and what comes out.

    A block that barely changes its input (score near 0) is doing little work.
    """
    _, _, layers = find_layers(model)
    n = len(layers)
    totals = torch.zeros(n, dtype=torch.float64)
    count = 0
    for c in chunks:
        x = c.unsqueeze(0).to(device_of(model))
        hs = model(input_ids=x, output_hidden_states=True, use_cache=False).hidden_states
        for i in range(min(n, len(hs) - 1)):
            a, b = hs[i][0].float(), hs[i + 1][0].float()
            totals[i] += (1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).sum().item()
        count += x.shape[1]
    return (totals / max(count, 1)).tolist()


def choose_layers_to_drop(scores: list[float], fraction: float) -> list[int]:
    n = len(scores)
    k = max(1, min(int(round(n * fraction)), n - 2))
    # Never drop the first or last block: they translate between tokens and
    # the model's internal language, and removing them is usually fatal.
    candidates = sorted(range(1, n - 1), key=lambda i: scores[i])
    return sorted(candidates[:k])


def drop_layers(model, drop: list[int]):
    parent, attr, layers = find_layers(model)
    drop_set = set(drop)
    kept_idx = [i for i in range(len(layers)) if i not in drop_set]
    kept = [layers[i] for i in kept_idx]
    setattr(parent, attr, nn.ModuleList(kept))
    # Attention layers remember their position, which the KV cache uses.
    for new_i, layer in enumerate(kept):
        for m in layer.modules():
            if isinstance(getattr(m, "layer_idx", None), int):
                m.layer_idx = new_i
    for cfg in {id(c): c for c in (model.config, getattr(model.config, "text_config", None)) if c is not None}.values():
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            if isinstance(getattr(cfg, key, None), int):
                setattr(cfg, key, len(kept))
        lt = getattr(cfg, "layer_types", None)
        if isinstance(lt, (list, tuple)) and len(lt) == len(layers):
            cfg.layer_types = [lt[i] for i in kept_idx]
    return kept_idx


def layer_prune(original, chunks, fraction: float = 0.25):
    scores = block_influence(original, chunks)
    drop = choose_layers_to_drop(scores, fraction)
    dieted = copy.deepcopy(original)
    drop_layers(dieted, drop)
    return dieted, {
        "plan": "layers",
        "label": PLANS["layers"]["label"],
        "fraction": fraction,
        "layers_before": len(scores),
        "layers_after": len(scores) - len(drop),
        "dropped_layers": drop,
        "block_influence": scores,
    }
