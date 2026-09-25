"""Run the seeded accuracy-vs-compute study for adaptive early-stopping.

For each seed we train one partially-reliable addition solver, draw a fixed pool
of ``N_MAX`` sampled chains per held-out question, and then replay that *same*
pool through every method - greedy, single-sample, fixed-N self-consistency and
adaptive early-stopping.  Because the chains are shared, differences in accuracy
and average compute are attributable purely to the decoding/voting policy.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from statistics import mean

import torch

from .consistency import adaptive_selfconsistency, fixed_selfconsistency
from .data import Example
from .model import count_parameters
from .solve import collect, pool_reliability
from .train import eval_examples, train_model

SEEDS = (0, 1, 2)
N_EVAL = 200
N_MAX = 32
TEMPERATURE = 1.0
TRAIN_STEPS = 450
FIXED_NS = (1, 2, 4, 8, 16, 32)
THETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
ADAPTIVE_REPORT_THETA = 0.9


def _eval_set(seed: int) -> list[Example]:
    return eval_examples(seed, N_EVAL)


@dataclass
class SeedMetrics:
    n_params: int
    greedy_acc: float
    single_acc: float                       # mean per-chain reliability
    fixed: dict[int, float] = field(default_factory=dict)         # n -> accuracy
    adaptive: dict[str, dict] = field(default_factory=dict)       # theta -> {acc, avg_n, early}
    adaptive_vs_fixed: dict[str, float] = field(default_factory=dict)  # theta -> disagree rate
    bucket_avg_n: dict[str, float] = field(default_factory=dict)  # reliability bucket -> avg chains


def _bucket(rel: float) -> str:
    if rel >= 1.0:
        return "consistent(1.0)"
    if rel >= 0.8:
        return "easy(0.8-1)"
    if rel >= 0.5:
        return "contested(.5-.8)"
    return "hard(<.5)"


def measure(seed: int) -> SeedMetrics:
    model = train_model(seed=seed, steps=TRAIN_STEPS)
    qs = collect(model, _eval_set(seed), N_MAX, TEMPERATURE)
    m = SeedMetrics(n_params=count_parameters(model), greedy_acc=0.0, single_acc=0.0)
    m.greedy_acc = round(sum(q.greedy == q.example.target for q in qs) / len(qs), 4)
    m.single_acc = round(mean(pool_reliability(q) for q in qs), 4)

    for n in FIXED_NS:
        acc = sum(fixed_selfconsistency(q.samples, n) == q.example.target for q in qs)
        m.fixed[n] = round(acc / len(qs), 4)

    for theta in THETAS:
        correct = 0
        total_n = 0
        early = 0
        disagree = 0
        full_ans = {i: fixed_selfconsistency(q.samples, N_MAX) for i, q in enumerate(qs)}
        for i, q in enumerate(qs):
            r = adaptive_selfconsistency(q.samples, N_MAX, theta)
            correct += r.answer == q.example.target
            total_n += r.n_used
            early += r.stopped_early
            disagree += r.answer != full_ans[i]
        m.adaptive[str(theta)] = {
            "accuracy": round(correct / len(qs), 4),
            "avg_chains": round(total_n / len(qs), 3),
            "early_stop_rate": round(early / len(qs), 4),
            "max_chains": N_MAX,
        }
        m.adaptive_vs_fixed[str(theta)] = round(disagree / len(qs), 4)

    # per-difficulty stopping behaviour at the headline theta
    bucket_n: dict[str, list[int]] = {}
    for q in qs:
        rel = pool_reliability(q)
        r = adaptive_selfconsistency(q.samples, N_MAX, ADAPTIVE_REPORT_THETA)
        bucket_n.setdefault(_bucket(rel), []).append(r.n_used)
    m.bucket_avg_n = {k: round(mean(v), 3) for k, v in bucket_n.items()}
    return m


def run_seed(seed: int) -> dict:
    return {"seed": seed, **asdict(measure(seed))}


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def environment() -> dict:
    """The machine these numbers came off, recorded beside them.

    Sampling chains and averaging logits over a batch are both float reductions whose
    order depends on the thread count and the torch build, so a rerun is bit-exact
    *inside* this environment and merely close outside it. The artifact says which
    one it is instead of the README claiming a reproducibility it cannot deliver.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def aggregate(per_seed: list[dict]) -> dict:
    out: dict = {"seeds": [s["seed"] for s in per_seed], "n_max": N_MAX,
                 "n_eval": N_EVAL, "report_theta": ADAPTIVE_REPORT_THETA}
    out["greedy_acc"] = {"mean": round(mean(s["greedy_acc"] for s in per_seed), 4),
                         "std": round(_std([s["greedy_acc"] for s in per_seed]), 4)}
    out["single_acc"] = {"mean": round(mean(s["single_acc"] for s in per_seed), 4),
                         "std": round(_std([s["single_acc"] for s in per_seed]), 4)}
    fixed = {}
    for n in FIXED_NS:
        vals = [s["fixed"][str(n)] if str(n) in s["fixed"] else s["fixed"][n]
                for s in per_seed]
        fixed[str(n)] = {"accuracy": round(mean(vals), 4),
                         "accuracy_std": round(_std(vals), 4)}
    out["fixed"] = fixed
    adaptive = {}
    for theta in THETAS:
        k = str(theta)
        acc = [s["adaptive"][k]["accuracy"] for s in per_seed]
        chains = [s["adaptive"][k]["avg_chains"] for s in per_seed]
        early = [s["adaptive"][k]["early_stop_rate"] for s in per_seed]
        dis = [s["adaptive_vs_fixed"][k] for s in per_seed]
        adaptive[k] = {"accuracy": round(mean(acc), 4),
                       "accuracy_std": round(_std(acc), 4),
                       "avg_chains": round(mean(chains), 3),
                       "early_stop_rate": round(mean(early), 4),
                       "disagree_rate_vs_fixed": round(mean(dis), 4)}
    out["adaptive"] = adaptive
    buckets: dict[str, list[float]] = {}
    for s in per_seed:
        for k, v in s["bucket_avg_n"].items():
            buckets.setdefault(k, []).append(v)
    out["bucket_avg_chains"] = {k: round(mean(v), 3) for k, v in buckets.items()}
    out["n_params_mean"] = round(mean(s["n_params"] for s in per_seed))
    return out


def build_results(per_seed: list[dict], runtime: float) -> dict:
    return {"config": {"seeds": list(SEEDS), "n_eval": N_EVAL, "n_max": N_MAX,
                       "temperature": TEMPERATURE, "train_steps": TRAIN_STEPS,
                       "fixed_ns": list(FIXED_NS), "thetas": list(THETAS)},
            "summary": aggregate(per_seed),
            "per_seed": per_seed,
            "environment": environment(),
            "runtime_sec": round(runtime, 1)}


__all__ = [
    "N_MAX",
    "SEEDS",
    "SeedMetrics",
    "aggregate",
    "build_results",
    "environment",
    "measure",
    "run_seed",
]
