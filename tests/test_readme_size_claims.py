"""The README's hand-written size claims must describe the code that exists.

Everything inside the RESULTS markers is byte-pinned against
``results/frontier.json``; the hand-written figures outside them - the "~700-line"
source size, the parameter count, the published runtime and the Quickstart's budget for
a rerun - are the numbers a reader takes on trust, so each one gets a guard here.
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


def test_readme_names_the_std_convention_the_tables_use():
    """`+/-` is ambiguous unless the file says which divisor produced it.

    The published spreads are the sample standard deviation over seeds, so the
    README has to use that word: a reader who recomputed the other convention would
    land on a different number and conclude the tables were wrong.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search("sample[^.]{0,60}standard\\s+deviation", readme), (
        "the README no longer states which standard-deviation convention its "
        "`+/-` columns use")


def test_the_published_wall_clock_is_the_one_the_artifact_records():
    """The rerun note pairs two runtimes and only the committed half of that is checkable."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    named = re.findall(r"published\s+(\d+(?:\.\d+)?)s(?![\d])", readme)
    assert named, "the rerun note no longer names the published runtime"
    assert len(named) == 1, f"the README names the published runtime more than once: {named}"
    artifact = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
    assert float(named[0]) == artifact["runtime_sec"], (
        f"README says the published run took {named[0]}s, "
        f"results/frontier.json records {artifact['runtime_sec']}s")


def test_the_quickstart_budget_matches_the_recorded_runtime():
    """The comment beside `run_study.py` is a budget a reader plans around."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    budgets = [int(m.group(1)) for m in re.finditer(r"\(~(\d+)s\)", readme)]
    assert budgets, "the Quickstart no longer budgets the study run; drop this guard with it"
    artifact = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
    for budget in budgets:
        assert 0.5 * budget <= artifact["runtime_sec"] <= 2.0 * budget, (
            f"the README budgets ~{budget}s for the study; the committed run took "
            f"{artifact['runtime_sec']}s")


def test_the_documented_rerun_writes_a_relative_scratch_file():
    """`--out /tmp/...` is not one path across shells, so the recipe must not use it."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--out /tmp/" not in readme, (
        "the rerun recipe is back to a /tmp path; Git-Bash rewrites it before the script "
        "sees it, so use a relative scratch file")
    assert "--out again-check.json" in readme, (
        "the rerun recipe no longer names the relative scratch file it documents")
