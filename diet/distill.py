"""Diet 3: Distillation, or "raising a leaner child".

The student starts as a copy of the teacher with half its layers removed (the
same Block Influence trick as layer pruning). Then it goes to the gym: it reads
text alongside the teacher and is trained to match the teacher's full
next-token probabilities, not just the right answer. That's what makes it
distillation rather than ordinary training.

Real distillation runs for days on billions of tokens. In the clinic it's a
short gym session, so treat the result as a taste of the method: watch the
loss fall and the student's agreement with its teacher recover.
"""

from __future__ import annotations

import copy
import random
import time

import torch
import torch.nn.functional as F

from .measure import compare, device_of
from .prune import block_influence, choose_layers_to_drop, drop_layers


def _batches(chunks, batch_size, steps, seed=0):
    rng = random.Random(seed)
    for _ in range(steps):
        picked = rng.sample(chunks, k=min(batch_size, len(chunks)))
        n = min(len(c) for c in picked)
        yield torch.stack([c[:n] for c in picked])


def distill(teacher, train_chunks, eval_chunks, keep_fraction=0.5, steps=80,
            batch_size=4, lr=1e-4, temperature=2.0, emit=lambda e: None):
    scores = block_influence(teacher, eval_chunks)
    drop = choose_layers_to_drop(scores, 1 - keep_fraction)
    student = copy.deepcopy(teacher)
    drop_layers(student, drop)

    emit({"type": "progress", "stage": "diet", "message": "Testing the training partner before any training…"})
    before_gym = compare(teacher, student, eval_chunks)

    final_dtype = next(student.parameters()).dtype
    student.float().train()
    teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    dev = device_of(student)
    curve = []
    t0 = time.perf_counter()
    for step, x in enumerate(_batches(train_chunks, batch_size, steps), start=1):
        x = x.to(dev)
        with torch.no_grad():
            t_logits = teacher(input_ids=x, use_cache=False).logits.float()
        s_logits = student(input_ids=x, use_cache=False).logits.float()
        V = s_logits.shape[-1]
        loss = F.kl_div(
            F.log_softmax(s_logits.view(-1, V) / temperature, -1),
            F.log_softmax(t_logits.view(-1, V) / temperature, -1),
            log_target=True,
            reduction="batchmean",
        ) * temperature**2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        curve.append(round(loss.item(), 5))
        emit({"type": "train", "step": step, "steps": steps, "loss": curve[-1]})
        del t_logits, s_logits, loss

    student.eval().to(final_dtype)
    del opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return student, {
        "plan": "student",
        "label": "Train a smaller partner",
        "keep_fraction": keep_fraction,
        "layers_before": len(scores),
        "layers_after": len(scores) - len(drop),
        "dropped_layers": drop,
        "block_influence": scores,
        "steps": steps,
        "batch_size": batch_size,
        "temperature": temperature,
        "train_seconds": time.perf_counter() - t0,
        "loss_curve": curve,
        "before_gym": {
            "agreement": before_gym["agreement"],
            "kl": before_gym["kl"],
            "perplexity": before_gym["perplexity_after"],
        },
    }
