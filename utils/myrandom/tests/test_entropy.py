"""Tests for non-deterministic seed and token helpers."""

import pytest

from utils.myrandom import fresh_root_seed, unique_token


def test_fresh_root_seed_has_requested_bit_bound():
    seed = fresh_root_seed(64)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**64


def test_unique_token_has_exact_hexadecimal_length():
    token = unique_token(12)
    assert len(token) == 24
    assert int(token, 16) >= 0


@pytest.mark.parametrize("value", [True, 1.5, "64"])
def test_fresh_root_seed_rejects_non_integer_bit_counts(value):
    with pytest.raises(TypeError):
        fresh_root_seed(value)


@pytest.mark.parametrize("value", [0, 24, 264, 65])
def test_fresh_root_seed_rejects_invalid_bit_counts(value):
    with pytest.raises(ValueError):
        fresh_root_seed(value)


@pytest.mark.parametrize("value", [False, 0, -1, 1.5, "8"])
def test_unique_token_rejects_invalid_byte_counts(value):
    expected = TypeError if not isinstance(value, int) or isinstance(value, bool) else ValueError
    with pytest.raises(expected):
        unique_token(value)
