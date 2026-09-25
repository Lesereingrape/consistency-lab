"""Supervised fine-tuning of the column-addition policy on a fixed difficulty band.

Training deliberately stops *short* of convergence so the solver stays partially
unreliable on 3-digit sums - per-question single-chain success lands broadly
between 0 and 1 rather than pinning at 1.0.  That partial reliability is the
precondition for self-consistency to add anything over a single sampled chain,
and for the adaptive stopping rule to have an accuracy-vs-compute trade to make.
"""

from __future__ import annotations

import random

import torch

from .data import PAD, Example
from .model import TinyTransformer

LO, HI = 100, 999


def train_model(seed: int = 0, steps: int = 450, batch: int = 64,
                lr: float = 3e-3, d_model: int = 48, n_layer: int = 2) -> TinyTransformer:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyTransformer(d_model=d_model, n_layer=n_layer)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        exs = [Example(rng.randint(LO, HI), rng.randint(LO, HI)) for _ in range(batch)]
        prompt = torch.tensor([e.prompt() for e in exs], dtype=torch.long)
        cot = torch.tensor([e.cot() for e in exs], dtype=torch.long)
        loss = model.sft_loss(prompt, cot)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def eval_examples(seed: int, n: int) -> list[Example]:
    """Held-out questions on a stream disjoint from the training draw."""
    rng = random.Random(900_000 + seed)
    return [Example(rng.randint(LO, HI), rng.randint(LO, HI)) for _ in range(n)]


__all__ = ["HI", "LO", "PAD", "eval_examples", "train_model"]
