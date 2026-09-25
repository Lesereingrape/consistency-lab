"""Command line entry point: watch one question go through the stopping rule.

    python -m conslab.cli demo
    python -m conslab.cli demo --theta 0.95 --seed 1

The demo draws its chains at the study's own temperature, chain cap and
stopping threshold, so what it prints is one question from the frontier the
README publishes rather than a differently-sampled look-alike.
"""

from __future__ import annotations

import argparse
from collections import Counter

from .consistency import adaptive_selfconsistency, fixed_selfconsistency, vote
from .model import count_parameters
from .solve import collect, pool_reliability
from .study import ADAPTIVE_REPORT_THETA, N_MAX, TEMPERATURE
from .train import eval_examples, train_model


def demo(seed: int = 0, theta: float = ADAPTIVE_REPORT_THETA, n_max: int = N_MAX,
         temperature: float = TEMPERATURE) -> None:
    model = train_model(seed=seed)
    print(f"seed {seed}  |  solver parameters = {count_parameters(model):,}")
    q = collect(model, eval_examples(seed, 8), n_max, temperature)[6]
    target = q.example.target
    from .data import render
    print(f"\nquestion: {render(q.example)}   true sum = {target}")
    print(f"greedy answer = {q.greedy} (correct={q.greedy == target})")
    rel = pool_reliability(q)
    print(f"single-chain reliability over {n_max} samples at temperature "
          f"{temperature} = {rel:.2f}\n")

    counts = Counter(a for a in q.samples if a is not None)
    print("answer distribution (value: count):")
    for val, c in counts.most_common(5):
        print(f"  {val}: {c}" + ("  <- correct" if val == target else ""))

    full = fixed_selfconsistency(q.samples, n_max)
    ar = adaptive_selfconsistency(q.samples, n_max, theta)
    print(f"\nfixed SC(N={n_max}) -> {full}  correct={full == target}")
    print(f"adaptive SC(θ={theta}) -> {ar.answer} in {ar.n_used} chains "
          f"(stopped_early={ar.stopped_early})  correct={ar.answer == target}")
    # replay where it decided
    running: Counter = Counter()
    for i, a in enumerate(q.samples, 1):
        running[a] += 1
        if i == ar.n_used:
            print(f"\nfirst {i} chains -> running plurality {vote(running)} "
                  f"(lead over runner-up decided at θ={theta})")
            break


def main() -> None:
    ap = argparse.ArgumentParser(prog="conslab")
    ap.add_argument("command", nargs="?", default="demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta", type=float, default=ADAPTIVE_REPORT_THETA)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = ap.parse_args()
    if args.command == "demo":
        demo(args.seed, args.theta, temperature=args.temperature)
    else:
        raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
