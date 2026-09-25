"""The carry-chain task is a lossless, exactly-verifiable encoding of a + b."""

from __future__ import annotations

import random

from conslab.data import (
    DIGIT0,
    NVOCAB,
    Example,
    parse_answer,
    render,
    sample_examples,
    score_answer,
    token_to_digit,
)


def test_cot_roundtrips_over_a_wide_range():
    rng = random.Random(0)
    for _ in range(3000):
        a, b = rng.randint(0, 999), rng.randint(0, 999)
        ex = Example(a, b)
        assert parse_answer(ex.cot(), ex) == a + b
        assert score_answer(ex.cot(), ex)


def test_parse_rejects_malformed_chain():
    ex = Example(123, 45)
    assert parse_answer([0, 0], ex) is None            # non-digit tokens
    assert parse_answer(ex.cot()[:-1], ex) is None     # truncated


def test_cot_is_causal_least_significant_first():
    # units column first: o_0 then its carry c_1
    ex = Example(95, 7)                                 # 95+7 = 102
    digs = [token_to_digit(t) for t in ex.cot()]
    assert digs[0] == 2                                 # units result digit
    assert digs[1] == 1                                 # carry into tens
    assert parse_answer(ex.cot(), ex) == 102


def test_sample_examples_respect_width():
    rng = random.Random(1)
    for ex in sample_examples(20, rng, 3):
        assert ex.ndigits == 3
        assert 100 <= ex.a <= 999 and 100 <= ex.b <= 999


def test_all_cot_tokens_are_digit_tokens():
    ex = Example(500, 814)
    assert all(DIGIT0 <= t < NVOCAB for t in ex.cot())


def test_render_is_readable_and_token_count_preserved():
    ex = Example(12, 30)
    text = render(ex, ex.cot())
    assert "+" in text and "=" in text
    assert len(text.split()) == len(ex.prompt()) + 2 * ex.ndigits
