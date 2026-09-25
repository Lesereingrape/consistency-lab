# consistency-lab — adaptive early-stopping self-consistency, measured on the accuracy-vs-compute frontier

**consistency-lab** is the "can I explore a frontier" flagship: a ~700-line, CPU-only
study that asks a genuinely open question about test-time compute and answers it with
measurement, not vibes. Standard *self-consistency* (Wang et al. 2022) draws a fixed
number N of chain-of-thought samples and majority-votes - wasteful when a question is
already decided after three samples. We implement an **adaptive stopping rule** that
halts sampling the moment the running vote is *statistically* decided, and we chart the
**accuracy-vs-compute frontier** it buys against fixed-N self-consistency on a verifiable
task a ~59k-parameter transformer can actually solve. Every number is reproduced on CPU.

![ci](https://github.com/Lesereingrape/consistency-lab/actions/workflows/ci.yml/badge.svg)

## The setup

- **A verifiable reasoning task:** 3-digit column addition emitted as an explicit
  *carry chain* (`o0 c1 o1 c2 …`), least-significant column first so it autoregresses
  causally. A Python oracle checks the reconstructed sum against `a+b` - the verifier
  never trusts the model's own arithmetic, so "correct" is ground truth.
- **A deliberately *partial* solver:** supervised fine-tuning stops short of
  convergence, so single-chain accuracy sits well below 1.0 and different temperature-
  sampled chains disagree. That spread is the raw material self-consistency exploits.
- **One shared pool, many policies:** per held-out question we pre-draw up to `N_MAX`
  sampled chains *once*. Greedy, single-sample, fixed-N self-consistency and the adaptive
  rule all replay that identical pool, so the only difference is **how many chains each
  spends** - which is exactly what a compute-limited deployment cares about.
- **The stopping rule:** treat the top-two answers as a Bernoulli split and halt as soon
  as a one-sided normal test says the leader's edge over 50% is real at confidence θ
  (cf. adaptive self-consistency, Aggarwal et al. 2023). No per-question oracle is used.

## Why it is trustworthy

Correctness is decided by an exact oracle, and every accuracy/chain figure is produced
by `experiments/run_study.py` and committed as `results/frontier.json`. The results
prose is rendered mechanically from that JSON by `experiments/make_report.py`, and a CI
test asserts the README equals it byte-for-byte - so numbers *and* the superlative
("which threshold wins the frontier") are computed from data and cannot be hand-tuned or
overclaimed. Notably we disclose the inconvenient result (greedy decoding is not just
near-optimal here, it beats the full-budget sampler) instead of hiding it.

The artifact also records the environment it was measured in (Python, torch, thread
count), and two runs in that environment reproduce it field for field - a test in
`tests/test_artifact_is_internally_consistent.py` recomputes the whole `summary` block
from the stored per-seed records. When a republished run moved the frontier by a couple
of points, the README was re-rendered from the new JSON rather than kept pretty.

## Quickstart

```bash
pip install -e .                    # torch is the only runtime dependency
python -m conslab.cli demo          # one question from the published pool: T=1.0, N<=32, theta=0.9
python experiments/run_study.py     # full 3-seed study -> results/frontier.json (~30s)
python experiments/run_study.py --out /tmp/again.json   # rerun elsewhere and diff it
python experiments/make_report.py --write   # splice the re-rendered block into README.md
```

## Results

<!-- RESULTS:START -->
*Every figure below is produced by `experiments/run_study.py` on CPU and committed as [`results/frontier.json`](results/frontier.json); the tables are rendered by `experiments/make_report.py`. One partially-trained ~59,053-parameter addition solver per seed draws the *same* pool of up to 32 temperature-sampled chains per held-out question; every policy below just chooses how many of those chains it spends. Mean over 3 seeds, 200 questions each.*

- task: 3-digit column addition as an explicit carry chain, graded by an exact `a+b` oracle (never the model's own arithmetic)
- under-training the solver leaves single-chain accuracy at **0.686** - genuinely between 0 and 1, so voting has real headroom
- fixed self-consistency at the full 32-chain budget = **0.953** (the accuracy the adaptive rule is chasing)
- measured under: Python 3.13.7 on Windows-11-10.0.26200-SP0, torch 2.14.0+cpu, 8 CPU threads, cpu - sampling and batched logits are float reductions whose order depends on that environment, so a rerun inside it is expected to be field-for-field identical and a rerun elsewhere only close

### Headline: the accuracy-vs-compute frontier

| decoding / voting policy | accuracy | mean chains/question |
|---|---:|---:|
| Single sampled chain (naive CoT) | 0.686 | 1 |
| Greedy (deterministic reference) | 0.982 | 1 |
| Fixed self-consistency, N=1 | 0.687 | 1 |
| Fixed self-consistency, N=2 | 0.665 | 2 |
| Fixed self-consistency, N=4 | 0.820 | 4 |
| Fixed self-consistency, N=8 | 0.932 | 8 |
| Fixed self-consistency, N=16 | 0.945 | 16 |
| Fixed self-consistency, N=32 | 0.953 | 32 |
| **Adaptive early-stop, θ=0.5** | 0.872 | 3.3 |
| **Adaptive early-stop, θ=0.6** | 0.872 | 3.3 |
| **Adaptive early-stop, θ=0.7** | 0.872 | 3.3 |
| **Adaptive early-stop, θ=0.8** | 0.947 | 5.0 |
| **Adaptive early-stop, θ=0.9** | 0.952 | 7.4 |
| **Adaptive early-stop, θ=0.95** | 0.952 | 7.9 |
| **Adaptive early-stop, θ=0.99** | 0.952 | 10.8 |

Adaptive early-stopping at **θ=0.8** reaches **0.947** using only **5.0 chains per question on average** - 6.4x cheaper than spending the full 32, and within seed noise of fixed-32-self-consistency (0.953). It also *dominates the fixed curve at equal cost*: at ~5.0 chains it scores 0.947 versus 0.820 for the most expensive fixed policy that fits in the same budget (N=4). The 100% early-stop rate and a mere 3.3% disagreement with the full-budget answer are the honest, measured sense in which the extra chains were wasted.

### The stopping rule reallocates compute by real difficulty

Binning questions by their true single-chain reliability (θ=0.9): *consistent(1.0)* **3.0**  *easy(0.8-1)* **4.2**  *contested(.5-.8)* **7.3**  *hard(<.5)* **14.6** mean chains. The rule *automatically* spends ~3 chains where the solver is unanimous and ~15 where it is contested, with no per-question oracle - that is the whole point of a sequential stopping test, and we measure it rather than assert it.

(The frontier headline above reports the *cheapest* threshold that still matches the full budget, θ=0.8; this difficulty split is measured at the published θ=0.9.)

### The honest catch: greedy is a strong reference here

On this deterministic arithmetic task greedy decoding already scores **0.982**, which *beats* full-budget self-consistency (0.953). That is a real, disclosed result and we will not hide it: when a single argmax chain is near-optimal, *any* sampling scheme pays to reproduce what greedy got for free. The value of adaptive stopping is therefore specifically the **sampled-chain frontier** (naive single chain 0.686 vs the voting policies above), which is the regime a stochastic hosted LLM actually lives in - and there the reduction above is genuine. We report the frontier, not a cherry-picked win.

### Honest limitations

- Deliberately toy: a ~59k-parameter carry-chain adder. Real self-consistency runs over free-form CoT on GSM8K-class problems; the *method* (a sequential stopping test over a shared sample pool, scored against an exact verifier) is what transfers, not this task.
- Seed-to-seed spread is large (single-chain std 0.17), because training a tiny adder to a fixed step count lands on different reliabilities; we show mean and std and read the frontier off the mean rather than tuning the step count to manufacture an SC-over-greedy gap.
- The stopping test is a one-sided normal approximation on the top-two answers; it is cheap and calibrates here, but θ is a knob we sweep, not a free lunch - very high θ erodes the savings toward the full budget.
<!-- RESULTS:END -->

## Why this is the frontier-exploration flagship

It is not a leaderboard chase - it is a controlled experiment about *test-time compute
allocation*, the resource that dominates real LLM serving cost. The reusable pieces (a
sequential statistical decision over a shared sample pool, scored by an exact verifier)
port directly to a hosted model; the toy adder is only there so the result is honest,
cheap and reproducible rather than asserted.
