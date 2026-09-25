"""Guard: the README RESULTS block must equal make_report.build(results).

Nothing in the published numbers is allowed to drift from the committed artifact.
If you re-run the study, regenerate the block; if you edit prose, do it in
make_report.py so this test keeps the README honest.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import make_report  # noqa: E402

MARKER = re.compile(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->", re.DOTALL)


def test_readme_matches_results():
    data = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = MARKER.search(readme)
    assert match, "README is missing the RESULTS markers"
    expected = make_report.build(data).strip()
    assert match.group(1).strip() == expected, (
        "README RESULTS block is stale; run `python experiments/make_report.py` "
        "and paste between the markers")
