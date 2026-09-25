"""The committed artifact must agree with itself.

``summary`` is a pure function of ``per_seed``, so recomputing it from the stored
per-seed records is a cheap end-to-end check that the published table is really the
average of the runs it claims to average - and that the environment the numbers were
taken in is recorded beside them rather than assumed.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from conslab.study import aggregate

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))


def test_summary_is_exactly_the_aggregate_of_the_stored_per_seed_runs():
    assert aggregate(DATA["per_seed"]) == DATA["summary"]


def test_per_seed_records_cover_the_published_seeds():
    assert [s["seed"] for s in DATA["per_seed"]] == DATA["config"]["seeds"]


def test_reported_std_is_the_sample_std_of_the_same_seed_runs():
    # ``aggregate`` uses the sample (n-1) standard deviation; the README says
    # "std", and a reader who computes numpy's default would not match it.
    summary = DATA["summary"]
    per_seed = DATA["per_seed"]
    assert summary["single_acc"]["std"] == round(
        statistics.stdev([s["single_acc"] for s in per_seed]), 4)
    assert summary["greedy_acc"]["std"] == round(
        statistics.stdev([s["greedy_acc"] for s in per_seed]), 4)
    for n, cell in summary["fixed"].items():
        assert cell["accuracy_std"] == round(
            statistics.stdev([s["fixed"][n] for s in per_seed]), 4), n
    for th, cell in summary["adaptive"].items():
        assert cell["accuracy_std"] == round(
            statistics.stdev([s["adaptive"][th]["accuracy"] for s in per_seed]), 4), th


def test_environment_block_is_recorded_with_the_numbers():
    env = DATA["environment"]
    for key in ("python", "platform", "torch", "threads", "device"):
        assert env[key], f"environment is missing {key!r}"
    assert env["device"] == "cpu"
    assert env["threads"] >= 1
