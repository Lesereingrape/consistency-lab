"""consistency-lab - adaptive early-stopping self-consistency for a tiny CoT model.

Instead of drawing a fixed number of reasoning samples and majority-voting
(self-consistency, Wang et al. 2022), a *sequential* stopping rule decides per
question when the running vote is already statistically decided and halts
sampling there.  The lab measures the honest accuracy-vs-compute frontier that
this buys, on a verifiable grade-school addition task a ~40k-parameter
transformer can actually solve - every number is reproduced on CPU.
"""

from __future__ import annotations

__all__ = ["cli", "consistency", "data", "model", "solve", "study", "train"]
