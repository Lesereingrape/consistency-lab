"""Sample collection returns a greedy answer plus a pool of parsed chains."""

from __future__ import annotations

from conslab.solve import collect, pool_reliability
from conslab.train import eval_examples, train_model


def test_collect_shapes_and_pool_size():
    model = train_model(seed=0, steps=120)
    exs = eval_examples(0, 10)
    qs = collect(model, exs, n_samples=12, temperature=1.0)
    assert len(qs) == len(exs)
    for q in qs:
        assert len(q.samples) == 12
        assert all(a is None or isinstance(a, int) for a in q.samples)
        assert q.greedy is None or isinstance(q.greedy, int)


def test_reliability_in_unit_interval():
    model = train_model(seed=1, steps=200)
    qs = collect(model, eval_examples(1, 20), n_samples=16)
    for q in qs:
        assert 0.0 <= pool_reliability(q) <= 1.0


def test_reliability_matches_counted_correct():
    model = train_model(seed=0, steps=120)
    q = collect(model, eval_examples(0, 5), n_samples=10)[0]
    target = q.example.target
    manual = sum(a == target for a in q.samples) / len(q.samples)
    assert abs(pool_reliability(q) - manual) < 1e-9
