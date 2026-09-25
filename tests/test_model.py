"""The tiny adder: forward shape, that SFT training learns, decoding works."""

from __future__ import annotations

import random

import torch

from conslab.data import Example
from conslab.model import TinyTransformer, count_parameters
from conslab.train import train_model


def _batch(n=16, seed=0):
    rng = random.Random(seed)
    exs = [Example(rng.randint(100, 999), rng.randint(100, 999)) for _ in range(n)]
    prompt = torch.tensor([e.prompt() for e in exs], dtype=torch.long)
    cot = torch.tensor([e.cot() for e in exs], dtype=torch.long)
    return exs, prompt, cot


def test_forward_shape():
    _exs, prompt, cot = _batch()
    model = TinyTransformer()
    x = torch.cat([prompt, cot[:, :-1]], dim=1)
    logits = model.forward(x)
    assert logits.shape[0] == x.shape[0] and logits.shape[2] == 13  # 3 special + 10 digits


def test_greedy_solver_is_accurate_on_training_width():
    model = train_model(seed=0, steps=450)
    exs, prompt, cot = _batch(24, seed=5)
    from conslab.data import parse_answer
    out = model.greedy_cot(prompt, cot.shape[1])
    got = [parse_answer(out[i].tolist(), exs[i]) for i in range(len(exs))]
    acc = sum(g == e.target for g, e in zip(got, exs, strict=True)) / len(exs)
    assert acc >= 0.8


def test_sampling_is_stochastic():
    model = train_model(seed=0, steps=300)
    _exs, prompt, cot = _batch(8, seed=3)
    a = model.sample_cot(prompt, cot.shape[1], temperature=1.0)
    b = model.sample_cot(prompt, cot.shape[1], temperature=1.0)
    assert not torch.equal(a, b)                        # different draws differ


def test_parameter_count_is_tiny():
    assert 20_000 < count_parameters(TinyTransformer()) < 120_000
