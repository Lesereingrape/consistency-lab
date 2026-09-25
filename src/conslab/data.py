"""Grade-school addition as a verifiable chain-of-thought task.

The model sees two operands and must emit the digit-by-digit column addition,
right-to-left, interleaving each result digit with the carry it produces::

    prompt:  a_{k-1} .. a0  +  b_{k-1} .. b0  =
    target:  o0 c1 o1 c2 o2 ... (result digit, then carry, per column)

Reading the result digits back gives the answer, which is checked *exactly*
against ``a + b`` by a Python oracle - the verifier never trusts the model's own
arithmetic.  Because each column is an independently error-prone step, the
model's per-question success rate is a real, measurable number strictly between
0 and 1, which is what makes both self-consistency and its early-stopping
variant meaningful here.

Difficulty is the number of digits: ``d=1`` is a single column, ``d=2`` two, and
so on.  The whole study stays on CPU because the model is tiny and a chain is a
handful of tokens.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PAD, PLUS, EQ = 0, 1, 2
DIGIT0 = 3
VOCAB = ["<pad>", "+", "="] + [str(d) for d in range(10)]
NVOCAB = len(VOCAB)


def digit_token(d: int) -> int:
    return DIGIT0 + d


def token_to_digit(t: int) -> int:
    return t - DIGIT0


def digits_of(n: int) -> list[int]:
    """Least-significant digit first."""
    return [int(c) for c in reversed(str(n))]


@dataclass(frozen=True)
class Example:
    a: int
    b: int

    @property
    def target(self) -> int:
        return self.a + self.b

    @property
    def ndigits(self) -> int:
        return max(len(digits_of(self.a)), len(digits_of(self.b)))

    def prompt(self) -> list[int]:
        ad, bd = digits_of(self.a), digits_of(self.b)
        d = self.ndigits
        ad = ad + [0] * (d - len(ad))
        bd = bd + [0] * (d - len(bd))
        # write most-significant first so the string reads naturally
        toks = [digit_token(x) for x in reversed(ad)]
        toks.append(PLUS)
        toks += [digit_token(x) for x in reversed(bd)]
        toks.append(EQ)
        return toks

    def cot(self) -> list[int]:
        """Ground-truth chain, least-significant column first.

        Emits ``o_0 c_1 o_1 c_2 ...``: the result digit of each column followed by
        the carry it produces.  Least-significant-first is the causal order - to
        predict column ``i``'s digit the model needs column ``i``'s incoming carry,
        which was just emitted as the previous token.
        """
        ad, bd = digits_of(self.a), digits_of(self.b)
        ncol = self.ndigits
        out: list[int] = []
        carry = 0
        for i in range(ncol):
            da = ad[i] if i < len(ad) else 0
            db = bd[i] if i < len(bd) else 0
            s = da + db + carry
            out.append(digit_token(s % 10))
            carry = s // 10
            out.append(digit_token(carry))
        return out


def sample_example(rng: random.Random, ndigits: int) -> Example:
    lo = 10 ** (ndigits - 1) if ndigits > 1 else 0
    hi = 10 ** ndigits - 1
    return Example(a=rng.randint(lo, hi), b=rng.randint(lo, hi))


def sample_examples(n: int, rng: random.Random, ndigits: int) -> list[Example]:
    return [sample_example(rng, ndigits) for _ in range(n)]


def parse_answer(cot_tokens: list[int], ex: Example) -> int | None:
    """Reconstruct the numeric sum from a generated CoT, or None if malformed.

    The chain stores ``o_0 c_1 o_1 c_2 ...`` (least-significant column first), so
    the result digits sit at the even positions.
    """
    if len(cot_tokens) < 2 * ex.ndigits:
        return None
    pairs = cot_tokens[: 2 * ex.ndigits]
    if any(not (DIGIT0 <= t <= DIGIT0 + 9) for t in pairs):
        return None
    digs = [token_to_digit(t) for t in pairs]          # o_0 c_1 o_1 c_2 ... o_{k-1} c_k
    total = sum(digs[2 * i] * (10 ** i) for i in range(ex.ndigits))
    total += digs[-1] * (10 ** ex.ndigits)             # the final carry-out
    return total


def score_answer(cot_tokens: list[int], ex: Example) -> bool:
    ans = parse_answer(cot_tokens, ex)
    return ans is not None and ans == ex.target


def render(ex: Example, cot_tokens: list[int] | None = None) -> str:
    toks = ex.prompt() + list(cot_tokens or [])
    return " ".join(VOCAB[t] for t in toks)
