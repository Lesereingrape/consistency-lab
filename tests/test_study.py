"""Aggregation and result assembly run on synthetic per-seed metric records."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from conslab.cli import demo
from conslab.study import ADAPTIVE_REPORT_THETA, FIXED_NS, SEEDS, THETAS, aggregate, build_results


def test_the_demo_draws_chains_at_the_published_settings():
    """A demo that samples differently is a second, unpublished experiment.

    ``demo`` used to call ``collect`` without a temperature and inherit the
    solver's 0.8 default, while the frontier was measured at 1.0. The printed
    single-chain reliability, fixed-SC and adaptive-SC answers therefore looked
    like the README's numbers and were not: same name, different sampler.
    """
    path = Path(__file__).resolve().parents[1] / "results" / "frontier.json"
    config = json.loads(path.read_text(encoding="utf-8"))["config"]
    defaults = {name: p.default for name, p in
                inspect.signature(demo).parameters.items()}
    assert defaults["temperature"] == config["temperature"]
    assert defaults["n_max"] == config["n_max"]
    assert defaults["theta"] == ADAPTIVE_REPORT_THETA
    assert defaults["theta"] in config["thetas"]


def _fake_seed(seed: int) -> dict:
    return {
        "seed": seed,
        "n_params": 59053,
        "greedy_acc": 0.9,
        "single_acc": 0.6,
        "fixed": {n: 0.6 + 0.05 * i for i, n in enumerate(FIXED_NS)},
        "adaptive": {str(t): {"accuracy": 0.85, "avg_chains": 3.0 + i,
                              "early_stop_rate": 0.9, "max_chains": 32}
                     for i, t in enumerate(THETAS)},
        "adaptive_vs_fixed": {str(t): 0.02 for t in THETAS},
        "bucket_avg_n": {"consistent(1.0)": 3.0, "hard(<.5)": 20.0},
    }


def test_fixed_across_seeds_is_key_tolerant():
    per = [_fake_seed(s) for s in SEEDS]
    agg = aggregate(per)
    assert agg["n_max"] == 32
    assert agg["fixed"]["1"]["accuracy"] < agg["fixed"][str(FIXED_NS[-1])]["accuracy"]


def test_aggregate_uses_report_theta():
    per = [_fake_seed(s) for s in SEEDS]
    agg = aggregate(per)
    assert str(ADAPTIVE_REPORT_THETA) in agg["adaptive"]
    assert agg["adaptive"][str(ADAPTIVE_REPORT_THETA)]["accuracy"] == 0.85
    assert agg["bucket_avg_chains"]["hard(<.5)"] == 20.0
    assert agg["n_params_mean"] == 59053


def test_build_results_wraps_config_and_runtime():
    per = [_fake_seed(s) for s in SEEDS]
    out = build_results(per, 8.4)
    assert out["config"]["seeds"] == list(SEEDS)
    assert out["config"]["n_max"] == 32
    assert out["runtime_sec"] == 8.4
    assert "summary" in out and len(out["per_seed"]) == len(SEEDS)


def test_std_of_constant_is_zero():
    per = [_fake_seed(s) for s in SEEDS]
    agg = aggregate(per)
    assert agg["single_acc"]["std"] == 0.0
