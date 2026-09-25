"""The stopping rule: decide early when unanimous, keep going when contested."""

from __future__ import annotations

from conslab.consistency import (
    adaptive_selfconsistency,
    fixed_selfconsistency,
    majority_of_samples_correct,
    vote,
)


def test_vote_plurality_and_tiebreak():
    assert vote([1, 1, 2, 3]) == 1
    assert vote([1, 2, 3]) == 1                        # tie -> smallest value
    assert vote([None, None]) is None                  # all malformed
    assert vote([None, 5, 5]) == 5                      # ignores malformed


def test_unanimous_pool_stops_at_minimum():
    samples = [7] * 32
    r = adaptive_selfconsistency(samples, n_max=32, theta=0.99, n_min=3)
    assert r.n_used == 3 and r.stopped_early and r.answer == 7


def test_contested_pool_runs_to_budget():
    # leader and runner-up stay neck-and-neck -> the test never fires
    samples = [1, 2] * 16
    r = adaptive_selfconsistency(samples, n_max=32, theta=0.99, n_min=3)
    assert r.used_all and not r.stopped_early and r.n_used == 32


def test_clear_lead_stops_early_and_matches_full_vote():
    samples = [1, 6, 7, 1, 1, 6, 1, 7, 1, 6, 1, 1]    # 1 takes a decided lead
    full = fixed_selfconsistency(samples, 12)
    r = adaptive_selfconsistency(samples, n_max=12, theta=0.8, n_min=3)
    assert r.stopped_early and r.n_used < 12
    assert r.answer == full == 1                        # early stop keeps the right answer


def test_theta_monotonicity():
    samples = [1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2]
    low = adaptive_selfconsistency(samples, 16, theta=0.6)
    high = adaptive_selfconsistency(samples, 16, theta=0.99)
    assert low.n_used <= high.n_used                    # looser test stops no later


def test_majority_correctness_helper():
    assert majority_of_samples_correct([5, 5, 4, 5], 5, 4) == 0.75
    assert majority_of_samples_correct([], 5, 4) == 0.0


def test_fixed_respects_budget_slice():
    samples = [1, 2, 2, 1, 1]
    assert fixed_selfconsistency(samples, 3) == 2        # only first 3 -> tie 1:2 -> value
    assert fixed_selfconsistency(samples, 5) == 1
