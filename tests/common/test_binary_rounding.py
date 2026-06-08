from fractions import Fraction

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from reformatters.common.binary_rounding import MANTISSA_BITS, round_float32_inplace


def test_round_float32_negative_values() -> None:
    values = np.linspace(-1000, -1, num=1000, dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=9)
    # All within 2% of the original value
    assert np.allclose(values, rounded_values, rtol=0.02)
    # Near zero bias introduced by rounding
    assert np.isclose(np.mean(values), np.mean(rounded_values))


def test_round_float32_positive_values() -> None:
    values = np.linspace(1, 1000, num=1000, dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=9)
    assert np.allclose(values, rounded_values, rtol=0.02)
    assert np.isclose(np.mean(values), np.mean(rounded_values))


def test_round_float32_negative_to_positive_values() -> None:
    values = np.linspace(-1e7, 1e7, num=2000, dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=9)
    assert np.allclose(values, rounded_values, rtol=0.02)
    assert np.isclose(np.mean(values), np.mean(rounded_values), rtol=0.02, atol=0.5)


def test_round_float32_special_values() -> None:
    special_values = np.array([0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    rounded_values = round_float32_inplace(special_values, keep_mantissa_bits=2)
    assert np.array_equal(special_values, rounded_values, equal_nan=True)


def test_round_float32_overflow_to_inf() -> None:
    # np.finfo(np.float32).max is the largest representable float32 value
    max_float32 = np.array([np.finfo(np.float32).max], dtype=np.float32)
    # With only 3 mantissa bits kept, this should overflow to infinity
    rounded_values = round_float32_inplace(max_float32, keep_mantissa_bits=3)
    assert np.all(np.isinf(rounded_values))


def test_round_float32_overflow_to_neg_inf() -> None:
    # np.finfo(np.float32).min is the most negative representable float32 value
    min_float32 = np.array([np.finfo(np.float32).min], dtype=np.float32)
    # With only 3 mantissa bits kept, this should overflow to negative infinity
    rounded_values = round_float32_inplace(min_float32, keep_mantissa_bits=3)
    assert np.all(np.isinf(rounded_values))
    assert np.all(rounded_values < 0)


def test_round_float32_mixed_overflow() -> None:
    # Create a float32 with all 1s in mantissa (23 bits)
    # Format: 0|00000000|11111111111111111111111
    # Sign: 0 (positive)
    # Exponent: 00000000 (0, leaving plenty of room for mantissa overflow)
    # Mantissa: All 1s - These all ones will guarantee the mantissa overflows when rounding
    bits = np.uint32(0b0_00000000_11111111111111111111111)
    mantissa_overflow = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]

    bits = np.uint32(0b0_00000000_0000000000000010110100)
    will_round_down = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    bits = np.uint32(0b0_00000000_00000000000000000000000)
    expected_rounded_down = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]

    # Create an array with one normal value and one that will overflow
    values = np.array(
        [mantissa_overflow, will_round_down, np.finfo(np.float32).max],
        dtype=np.float32,
    )

    rounded_values = round_float32_inplace(values, keep_mantissa_bits=4)

    # Check that the normal value remains approximately the same
    assert np.isclose(rounded_values[0], mantissa_overflow)
    # But not exactly the same (it's rounded)
    assert rounded_values[0] != mantissa_overflow

    # Check the value that should round down, did
    assert rounded_values[1] == expected_rounded_down

    # Check that the large value overflowed to infinity
    assert np.isinf(rounded_values[2])
    assert rounded_values[2] > 0  # Positive infinity


def test_round_up_no_tie_positive() -> None:
    # Create a float32 with specific bit pattern
    # Format: 0|10000000|10101010101010101010101
    # Sign: 0 (positive)
    # Exponent: 10000000
    # Mantissa: 10101010101010101010101
    bits = np.uint32(0b0_10000000_10101010101010101010101)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 3.3333333

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=4)

    # Should round up
    assert rounded_values[0] > original

    expected_bits = np.uint32(0b0_10000000_10110000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected
    assert rounded_values[0] == 3.375


def test_round_up_no_tie_negative() -> None:
    # Format: 1|10000000|10101010101010101010101
    bits = np.uint32(0b1_10000000_10101010101010101010101)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == -3.3333333

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=4)

    expected_bits = np.uint32(0b1_10000000_10110000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected
    assert rounded_values[0] == -3.375

    # "up" means away from zero for negative numbers
    assert rounded_values[0] < original

    # Check that rounding down would have been further from original
    rounded_down_bits = np.uint32(0b1_10000000_10100000000000000000000)
    rounded_down = np.frombuffer(rounded_down_bits.tobytes(), dtype=np.float32)[0]
    assert abs(original - expected) < abs(original - rounded_down)


def test_round_down_no_tie_positive() -> None:
    # Format: 0|10000011|10101010101010101010101
    bits = np.uint32(0b0_10000011_10101010101010101010101)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 26.666666

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=5)

    expected_bits = np.uint32(0b0_10000011_10101000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected
    assert rounded_values[0] == 26.5

    assert rounded_values[0] < original

    # Check that rounding up would have been further from original
    rounded_up_bits = np.uint32(0b0_10000011_10111000000000000000000)
    rounded_up = np.frombuffer(rounded_up_bits.tobytes(), dtype=np.float32)[0]
    assert abs(original - expected) < abs(original - rounded_up)


def test_round_down_no_tie_negative() -> None:
    # Format: 1|10000011|10101010101010101010101
    bits = np.uint32(0b1_10000011_10101010101010101010101)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == -26.666666

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=5)

    expected_bits = np.uint32(0b1_10000011_10101000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected
    assert rounded_values[0] == -26.5

    assert rounded_values[0] > original

    # Check that rounding up would have been further from original
    rounded_up_bits = np.uint32(0b1_10000011_10111000000000000000000)
    rounded_up = np.frombuffer(rounded_up_bits.tobytes(), dtype=np.float32)[0]
    assert abs(original - expected) < abs(original - rounded_up)


def test_round_tie_down_to_even_positive() -> None:
    # Format: 0|10000010|00100000000000000000000
    bits = np.uint32(0b0_10000010_00100000000000000000000)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 9.0

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=2)

    expected_bits = np.uint32(0b0_10000010_00000000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected
    assert rounded_values[0] == 8.0

    assert rounded_values[0] < original

    # Check equidistant rounding
    rounded_up_bits = np.uint32(0b0_10000010_01000000000000000000000)
    rounded_up = np.frombuffer(rounded_up_bits.tobytes(), dtype=np.float32)[0]
    assert abs(original - expected) == abs(original - rounded_up)


def test_round_tie_up_to_even_positive() -> None:
    # Format: 0|10000010|00110000000000000000000
    bits = np.uint32(0b0_10000010_00110000000000000000000)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 9.5

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=3)

    expected_bits = np.uint32(0b0_10000010_01000000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected
    assert rounded_values[0] == 10.0

    assert rounded_values[0] > original

    # Check equidistant rounding
    rounded_down_bits = np.uint32(0b0_10000010_00100000000000000000000)
    rounded_down = np.frombuffer(rounded_down_bits.tobytes(), dtype=np.float32)[0]
    assert abs(original - expected) == abs(original - rounded_down)


def test_round_keep_all_bits() -> None:
    # Format: 0|10000001|10000000000000000000111
    bits = np.uint32(0b0_10000001_10000000000000000000111)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 6.0000033

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(
        values, keep_mantissa_bits=23
    )  # Keep all bits

    assert rounded_values[0] == original


def test_round_keep_zero_bits() -> None:
    # Format: 0|10000001|10000000000000000000111
    bits = np.uint32(0b0_10000001_10000000000000000000000)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 6.0

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=0)

    expected_bits = np.uint32(0b0_10000010_00000000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected


def test_round_keep_all_but_one_bits_trailing_1() -> None:
    # Format: 0|10000001|10000000000000000000111
    bits = np.uint32(0b0_10000001_10000000000000000000111)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 6.0000033

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=22)

    expected_bits = np.uint32(0b0_10000001_10000000000000000001000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected


def test_round_keep_all_but_one_bits_trailing_0() -> None:
    # Format: 0|10000001|10000000000000000000110
    bits = np.uint32(0b0_10000001_10000000000000000000110)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 6.000003

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=22)

    expected_bits = np.uint32(0b0_10000001_10000000000000000000110)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected


def test_round_mantissa_overflow() -> None:
    # Format: 0|10000001|11111111111111111111111
    bits = np.uint32(0b0_10000001_11111111111111111111111)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 7.9999995

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=15)

    expected_bits = np.uint32(0b0_10000010_00000000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert rounded_values[0] == expected
    assert rounded_values[0] == 8.0


def test_round_exponent_and_mantissa_overflow_positive() -> None:
    # Format: 0|11111110|11111111111111111111111
    bits = np.uint32(0b0_11111110_11111111111111111111111)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=20)

    assert np.isinf(rounded_values[0])
    assert rounded_values[0] > 0


def test_round_exponent_and_mantissa_overflow_negative() -> None:
    # Format: 1|11111110|11111111111111111111111
    bits = np.uint32(0b1_11111110_11111111111111111111111)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=20)

    assert np.isinf(rounded_values[0])
    assert rounded_values[0] < 0


def test_round_zero() -> None:
    values = np.array([0.0], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=20)
    assert rounded_values[0] == 0.0


def test_round_subnormal_to_normal() -> None:
    # Format: 0|00000000|11111111111111111111111
    bits = np.uint32(0b0_00000000_11111111111111111111111)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 1.1754942e-38

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=20)

    expected_bits = np.uint32(0b0_00000001_00000000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert expected == 1.1754944e-38
    assert rounded_values[0] == expected


def test_round_subnormal_to_subnormal() -> None:
    # Format: 0|00000000|00011111111111111111111
    bits = np.uint32(0b0_00000000_00011111111111111111111)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 1.469367e-39

    values = np.array([original], dtype=np.float32)
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=3)

    expected_bits = np.uint32(0b0_00000000_00100000000000000000000)
    expected = np.frombuffer(expected_bits.tobytes(), dtype=np.float32)[0]
    assert expected == 1.469368e-39
    assert rounded_values[0] == expected


def test_round_subnormal_to_zero() -> None:
    # Format: 0|00000000|00000000000000000000001
    # Smallest possible subnormal number
    bits = np.uint32(0b0_00000000_00000000000000000000001)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]
    assert original == 1.4e-45  # Smallest possible subnormal float32

    values = np.array([original], dtype=np.float32)
    # Keep only a few bits - this should round to zero since all significant bits will be dropped
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=3)
    assert rounded_values[0] == 0.0

    # Test with a slightly larger subnormal number
    # Format: 0|00000000|00000000000000000001111
    bits = np.uint32(0b0_00000000_00000000000000000001111)
    original = np.frombuffer(bits.tobytes(), dtype=np.float32)[0]

    values = np.array([original], dtype=np.float32)
    # This should also round to zero when keeping only a few bits
    rounded_values = round_float32_inplace(values, keep_mantissa_bits=5)
    assert rounded_values[0] == 0.0


def test_wide_logspace_percent_difference() -> None:
    values = np.logspace(-127, 127, num=2000, base=2, dtype=np.float32)
    original = values.copy()
    rounded = round_float32_inplace(values, keep_mantissa_bits=9)
    diff_percent = np.abs((original - rounded) / original) * 100
    max_diff = np.max(diff_percent)

    assert max_diff > 0.0  # confirm that we have actually rounded the value
    assert max_diff < 0.5  # Less than 1/2 of 1% error


# ---------------------------------------------------------------------------
# Property-based tests
#
# Invariants exercised below:
#   1. Matches an independent round-half-to-even reference (rational arithmetic).
#   2. Special values (+/-0, +/-inf, NaN) are preserved.
#   3. Idempotent: rounding an already-rounded value is a no-op.
#   4. Order preserving (monotonic non-decreasing).
#   5. Sign symmetric: round(-x) == -round(x).
#   6. Finite results keep zeros in the dropped low mantissa bits.
#   7. Relative error is bounded by 2**-keep_mantissa_bits for normal numbers.
# ---------------------------------------------------------------------------

_INF_BITS = np.uint32(0x7F800000)
_SIGN_BIT = np.uint32(0x80000000)
_SMALLEST_NORMAL_F32 = float(np.finfo(np.float32).tiny)


def _round_reference(value: np.float32, keep_mantissa_bits: int) -> np.float32:
    """Independent round-half-to-even reference.

    Among the float32 values whose low ``23 - keep`` mantissa bits are zero, pick
    the one nearest ``value``. Ties go to the candidate whose bit at the kept-LSB
    position is zero (round to even). The nearest/tie decision is made with exact
    rational arithmetic, so it does not reuse the bit logic under test.
    """
    if keep_mantissa_bits == MANTISSA_BITS or not np.isfinite(value) or value == 0:
        return value

    drop = MANTISSA_BITS - keep_mantissa_bits
    bits = int(np.float32(value).view(np.uint32))
    sign = bits & int(_SIGN_BIT)
    magnitude = bits & 0x7FFFFFFF
    step = 1 << drop

    down = magnitude - (magnitude % step)  # truncate toward zero in magnitude
    up = down + step

    x = abs(Fraction(float(value)))
    down_value = Fraction(float(np.uint32(down).view(np.float32)))
    up_is_inf = up >= int(_INF_BITS)
    if up_is_inf:
        up_value = Fraction(2) ** 128  # hypothetical value one ULP past max float32
    else:
        up_value = Fraction(float(np.uint32(up).view(np.float32)))

    dist_down = x - down_value
    dist_up = up_value - x
    if dist_up < dist_down:
        choose_up = True
    elif dist_up > dist_down:
        choose_up = False
    else:  # exact tie -> round to even (the candidate whose kept LSB is zero)
        choose_up = ((down >> drop) & 1) == 1

    if choose_up and up_is_inf:
        return np.float32(np.inf if sign == 0 else -np.inf)

    result_bits = (up if choose_up else down) | sign
    return np.uint32(result_bits).view(np.float32)


@settings(deadline=None, max_examples=500)
@given(
    value=st.floats(width=32, allow_nan=False, allow_infinity=False),
    keep=st.integers(min_value=0, max_value=MANTISSA_BITS),
)
def test_round_matches_independent_reference(value: float, keep: int) -> None:
    result = round_float32_inplace(np.array([value], dtype=np.float32), keep)[0]
    expected = _round_reference(np.float32(value), keep)
    # Both are float32; compare bit patterns to also catch signed-zero differences.
    assert result.view(np.uint32) == np.float32(expected).view(np.uint32), (
        f"value={value!r} keep={keep}: got {result!r}, expected {expected!r}"
    )


@settings(deadline=None, max_examples=200)
@given(keep=st.integers(min_value=0, max_value=MANTISSA_BITS))
def test_round_preserves_special_values(keep: int) -> None:
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, -np.nan], dtype=np.float32)
    result = round_float32_inplace(special.copy(), keep)
    np.testing.assert_array_equal(result, special)  # treats NaN==NaN
    # signed zero sign bit must be preserved exactly
    assert result[0].view(np.uint32) == np.uint32(0)
    assert result[1].view(np.uint32) == _SIGN_BIT


@settings(deadline=None, max_examples=300)
@given(
    arr=arrays(
        np.float32,
        st.integers(min_value=1, max_value=64),
        elements=st.floats(width=32),
    ),
    keep=st.integers(min_value=0, max_value=MANTISSA_BITS),
)
def test_round_is_idempotent(arr: np.ndarray, keep: int) -> None:
    once = round_float32_inplace(arr.copy(), keep)
    twice = round_float32_inplace(once.copy(), keep)
    np.testing.assert_array_equal(once, twice)


@settings(deadline=None, max_examples=300)
@given(
    arr=arrays(
        np.float32,
        st.integers(min_value=2, max_value=64),
        elements=st.floats(width=32, allow_nan=False),
    ),
    keep=st.integers(min_value=0, max_value=MANTISSA_BITS),
)
def test_round_is_monotonic(arr: np.ndarray, keep: int) -> None:
    ascending = np.sort(arr)  # places -inf first, +inf last, no NaN present
    rounded = round_float32_inplace(ascending.copy(), keep)
    assert np.all(rounded[:-1] <= rounded[1:])


@settings(deadline=None, max_examples=500)
@given(
    value=st.floats(width=32, allow_nan=False),
    keep=st.integers(min_value=0, max_value=MANTISSA_BITS),
)
def test_round_is_sign_symmetric(value: float, keep: int) -> None:
    pos = round_float32_inplace(np.array([value], dtype=np.float32), keep)[0]
    neg = round_float32_inplace(np.array([-value], dtype=np.float32), keep)[0]
    # round(-x) == -round(x), including for +/-inf overflow
    assert (-neg).view(np.uint32) == pos.view(np.uint32)


@settings(deadline=None, max_examples=500)
@given(
    value=st.floats(width=32, allow_nan=False, allow_infinity=False),
    keep=st.integers(min_value=0, max_value=MANTISSA_BITS - 1),
)
def test_round_clears_dropped_mantissa_bits(value: float, keep: int) -> None:
    result = round_float32_inplace(np.array([value], dtype=np.float32), keep)[0]
    if not np.isfinite(result):
        return  # overflow to inf is allowed
    drop = MANTISSA_BITS - keep
    low_mantissa_bits = int(result.view(np.uint32)) & ((1 << drop) - 1)
    assert low_mantissa_bits == 0


@settings(deadline=None, max_examples=500)
@given(
    value=st.floats(
        min_value=_SMALLEST_NORMAL_F32, max_value=2.0**100, width=32, allow_nan=False
    ),
    keep=st.integers(min_value=1, max_value=MANTISSA_BITS),
)
def test_round_relative_error_bounded(value: float, keep: int) -> None:
    result = round_float32_inplace(np.array([value], dtype=np.float32), keep)[0]
    if not np.isfinite(result):
        return
    relative_error = abs(float(result) - value) / value
    assert relative_error <= 2.0**-keep
