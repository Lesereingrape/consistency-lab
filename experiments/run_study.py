"""Run the full 3-seed accuracy-vs-compute study and write results/frontier.json.

    python experiments/run_study.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from conslab.study import SEEDS, build_results, run_seed


def main() -> None:
    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]
    results = build_results(per_seed, time.time() - t0)
    out = Path("results/frontier.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    s = results["summary"]
    print(f"wrote {out} in {results['runtime_sec']}s")
    print(f"single={s['single_acc']['mean']:.3f} "
          f"fixed32={s['fixed']['32']['accuracy']:.3f} "
          f"adaptive@0.9: acc={s['adaptive']['0.9']['accuracy']:.3f} "
          f"chains={s['adaptive']['0.9']['avg_chains']:.1f}")


if __name__ == "__main__":
    main()
