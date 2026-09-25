"""Run the full 3-seed accuracy-vs-compute study and write results/frontier.json.

    python experiments/run_study.py [--out PATH]

``--out`` exists so a second run can be written somewhere else and compared field
for field against the committed artifact, which is how the README's reproducibility
sentence gets checked rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from conslab.study import SEEDS, build_results, run_seed


def main(out: str = "results/frontier.json") -> None:
    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]
    results = build_results(per_seed, time.time() - t0)
    dest = Path(out)
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    s = results["summary"]
    print(f"wrote {dest} in {results['runtime_sec']}s")
    print(f"single={s['single_acc']['mean']:.3f} "
          f"fixed32={s['fixed']['32']['accuracy']:.3f} "
          f"adaptive@0.9: acc={s['adaptive']['0.9']['accuracy']:.3f} "
          f"chains={s['adaptive']['0.9']['avg_chains']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="run_study")
    ap.add_argument("--out", default="results/frontier.json",
                    help="where to write the artifact (default: results/frontier.json)")
    main(ap.parse_args().out)

