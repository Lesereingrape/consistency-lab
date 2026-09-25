"""The README's hand-written size claims must describe the code that exists.

Everything inside the RESULTS markers is byte-pinned against
``results/frontier.json``; the "~700-line" figure in the first paragraph is the one
other hand-written number a reader takes on trust, so it gets a guard too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source_lines() -> int:
    return sum(len(p.read_text(encoding="utf-8").splitlines())
               for p in sorted((ROOT / "src").rglob("*.py")))


def test_readme_line_count_claim_matches_the_source():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"a ~(\d+)-line", readme)
    assert m, "README no longer states its source size; drop or restore the claim"
    claimed = int(m.group(1))
    actual = _source_lines()
    assert abs(claimed - actual) <= 50, (
        f"README says ~{claimed} lines, src/ has {actual}; update the claim")


def test_readme_parameter_claim_matches_the_artifact():
    data = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
    mean_params = data["summary"]["n_params_mean"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for m in re.finditer(r"a ~(\d+)k-parameter", readme):
        assert abs(int(m.group(1)) * 1000 - mean_params) <= 1000, (
            f"README says ~{m.group(1)}k parameters, the artifact averages "
            f"{mean_params:,}")
