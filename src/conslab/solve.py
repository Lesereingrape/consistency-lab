"""Draw reasoning samples from the policy for a batch of questions.

For every question we record the greedy single answer (the k=1 baseline) and a
fixed pool of ``n_samples`` temperature-sampled answers.  The self-consistency
and adaptive-stopping logic in ``consistency.py`` then operates purely on that
recorded pool, so every method sees the *exact same* samples - the only thing
that differs between them is how many of the pool they actually consume.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from .data import Example, parse_answer
from .model import TinyTransformer


@dataclass
class QuestionSamples:
    example: Example
    greedy: int | None
    samples: list[int | None]       # one parsed answer per sampled chain


def _answers_from(cot: torch.Tensor, exs: list[Example]) -> list[int | None]:
    return [parse_answer(cot[i].tolist(), exs[i]) for i in range(len(exs))]


def collect(model: TinyTransformer, examples: list[Example], n_samples: int,
            temperature: float = 0.8) -> list[QuestionSamples]:
    """Sample ``n_samples`` chains per question, grouped so prompt widths stay uniform."""
    model.eval()
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        groups[ex.ndigits].append(idx)

    out: list[QuestionSamples | None] = [None] * len(examples)
    with torch.no_grad():
        for nd, idxs in groups.items():
            exs = [examples[i] for i in idxs]
            prompt = torch.tensor([e.prompt() for e in exs], dtype=torch.long)
            greedy = model.greedy_cot(prompt, 2 * nd)
            g_ans = _answers_from(greedy, exs)

            tiled = prompt.repeat_interleave(n_samples, dim=0)
            cot = model.sample_cot(tiled, 2 * nd, temperature)
            cot = cot.view(len(exs), n_samples, 2 * nd)
            for bi, i in enumerate(idxs):
                raw = cot[bi].reshape(-1).tolist()
                per = [parse_answer(raw[s * 2 * nd:(s + 1) * 2 * nd], examples[i])
                       for s in range(n_samples)]
                out[i] = QuestionSamples(examples[i], g_ans[bi], per)
    return [q for q in out if q is not None]


def is_correct(pool: list[int | None], target: int) -> list[bool]:
    return [a == target for a in pool]


def pool_reliability(q: QuestionSamples) -> float:
    """Empirical single-sample success rate for one question."""
    n = len(q.samples)
    return sum(a == q.example.target for a in q.samples) / n if n else 0.0
