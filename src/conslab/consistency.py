"""Self-consistency and its adaptive early-stopping variant.

Both consume the *same* pool of sampled chains per question and agree on the
answer by majority vote; they differ only in how many chains they look at:

- ``fixed_selfconsistency``  - classic self-consistency: always spend all ``n_max``
  chains, vote over them (Wang et al. 2022, arXiv:2203.11171).
- ``adaptive_selfconsistency`` - a sequential stopping rule: draw chains one at a
  time and halt as soon as the running plurality is *statistically decided*, i.e.
  the lead over the runner-up is wide enough that the remaining budget is very
  unlikely to overturn it.  This is the efficiency contribution measured here
  (cf. adaptive self-consistency, Aggarwal et al. 2023, arXiv:2305.11860).

The stopping test treats the top-two answers as a Bernoulli split and asks, via a
one-sided normal test on the observed share, whether the leader's edge over 50%
is real at confidence ``theta``.  Questions the model answers consistently stop
after a couple of chains; genuinely contested ones keep going to ``n_max``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _as_counts(x) -> Counter:
    """Accept either an iterable of answers or an already-tallied Counter."""
    return x if isinstance(x, (Counter, dict)) else Counter(x)


def vote(counts) -> int | None:
    """Plurality answer (ignoring malformed) with a deterministic value tie-break."""
    answered = {a: c for a, c in _as_counts(counts).items() if a is not None}
    if not answered:
        return None
    best = max(answered.values())
    return min(a for a, c in answered.items() if c == best)


def fixed_selfconsistency(samples: list[int | None], n_max: int) -> int | None:
    return vote(samples[:n_max])


@dataclass
class AdaptiveResult:
    answer: int | None
    n_used: int          # chains actually consumed (1-indexed count)
    stopped_early: bool
    used_all: bool       # ran to the n_max budget without deciding


def adaptive_selfconsistency(samples: list[int | None], n_max: int,
                             theta: float, n_min: int = 3) -> AdaptiveResult:
    """Sequentially consume chains, halting once the vote is decided at ``theta``.

    ``samples`` is the pre-drawn pool; we only *read* up to ``n_used`` of them, so
    the compute actually spent is ``n_used`` chains, not ``n_max``.
    """
    counts: Counter = Counter()
    pool = samples[:n_max]
    for i, a in enumerate(pool, start=1):
        counts[a] += 1
        if i >= n_min and a is not None:
            answered = {k: v for k, v in counts.items() if k is not None}
            if len(answered) >= 2:
                c1, c2 = sorted(answered.values(), reverse=True)[:2]
            else:
                c1, c2 = answered.get(a, 0), 0
            n = c1 + c2
            if c1 > c2 and n >= n_min:
                p_hat = c1 / n
                std = math.sqrt(p_hat * (1 - p_hat) / n)
                z = (p_hat - 0.5) / std if std > 0 else math.inf
                if _phi(z) >= theta:
                    return AdaptiveResult(vote(counts), i, True, False)
    return AdaptiveResult(vote(counts), len(pool), False, True)


def majority_of_samples_correct(samples: list[int | None], target: int,
                                n_max: int) -> float:
    """Per-question empirical single-chain success rate over the pool."""
    pool = samples[:n_max]
    if not pool:
        return 0.0
    return sum(a == target for a in pool) / len(pool)
