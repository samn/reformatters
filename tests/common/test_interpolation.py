import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from reformatters.common.interpolation import linear_interpolate_1d_inplace


def test_linear_interpolate_1d_inplace_every_other_point() -> None:
    # Create a 3D data array with linearly increasing values along first dimension
    z = np.arange(5)
    y = np.arange(3)
    x = np.arange(2)
    data = np.zeros((len(z), len(y), len(x)), dtype=np.float32)
    for i in range(len(y)):
        for j in range(len(x)):
            data[:, i, j] = np.linspace(i * 10 + j, (i + 1) * 10 + j, len(z))

    da = xr.DataArray(data, coords=[("z", z), ("y", y), ("x", x)])

    orig_da = da.copy(deep=True)

    da.values[1::2, :, :] = np.nan

    result = linear_interpolate_1d_inplace(
        da, dim="z", where=np.isnan(da.values[:, 0, 0])
    )

    np.testing.assert_allclose(result.values, orig_da.values)
    assert result is da


def test_linear_interpolate_1d_inplace_where_all_false() -> None:
    # Create a 3D data array with linearly increasing values along first dimension
    z = np.arange(5)
    y = np.arange(3)
    x = np.arange(2)
    data = np.zeros((len(z), len(y), len(x)), dtype=np.float32)
    for i in range(len(y)):
        for j in range(len(x)):
            data[:, i, j] = np.linspace(i * 10 + j, (i + 1) * 10 + j, len(z))

    da = xr.DataArray(data, coords=[("z", z), ("y", y), ("x", x)])

    orig_da = da.copy(deep=True)

    da.values[1::2, :, :] = np.nan

    values_copy = da.values.copy()

    result = linear_interpolate_1d_inplace(
        da, dim="z", where=np.zeros(da.shape[0], dtype=bool)
    )

    np.testing.assert_allclose(result.values, values_copy)
    assert not np.allclose(result.values, orig_da.values)
    assert result is da


def test_linear_interpolate_1d_inplace_unexpected_nan() -> None:
    data = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    da = xr.DataArray(data, dims=["z", "y", "x"])

    da.values[::2, :, :] = np.nan
    da.values[1, :, :] = np.nan  # "unexpected" nan

    # First True has unexpected nan before it so interpolation returns nan
    # Second True has valid surrounding values so interpolation returns a value (4)
    where = np.array([False, False, True, False, True, False])
    result = linear_interpolate_1d_inplace(da, dim="z", where=where)

    np.testing.assert_equal(
        result.values.ravel(),
        np.array([np.nan, np.nan, np.nan, 3, 4, 5], dtype=np.float32),
    )
    assert result is da


# ---------------------------------------------------------------------------
# Property-based tests
#
# Invariants exercised below:
#   1. Interpolated points equal the midpoint of their two original neighbors.
#   2. Non-selected points and the endpoints are left unchanged.
#   3. A linear sequence is recovered exactly when its interior points are
#      interpolated (linear interpolation reproduces a line).
#   4. A NaN neighbor propagates to NaN.
#   5. Idempotent: a second pass leaves the array unchanged.
#   6. Consecutive selected points are rejected.
#   7. Returns the same DataArray object (mutated in place).
# ---------------------------------------------------------------------------


def _non_consecutive_mask(draw_bools: list[bool]) -> np.ndarray:
    """Clear any True that immediately follows another True, satisfying the
    'no consecutive interpolation' contract while keeping a varied mask."""
    mask = np.array(draw_bools, dtype=bool)
    for i in range(1, len(mask)):
        if mask[i] and mask[i - 1]:
            mask[i] = False
    return mask


_lengths = st.integers(min_value=3, max_value=40)
_finite_f32 = st.floats(
    min_value=-1e6, max_value=1e6, width=32, allow_nan=False, allow_infinity=False
)


@settings(deadline=None, max_examples=300)
@given(data=st.data())
def test_interpolate_equals_neighbor_midpoint(data: st.DataObject) -> None:
    n = data.draw(_lengths)
    rows = data.draw(st.integers(min_value=1, max_value=4))
    cols = data.draw(st.integers(min_value=1, max_value=4))
    values = data.draw(arrays(np.float32, (n, rows, cols), elements=_finite_f32))
    where = _non_consecutive_mask(
        data.draw(st.lists(st.booleans(), min_size=n, max_size=n))
    )

    original = values.copy()
    da = xr.DataArray(values, dims=["z", "y", "x"])
    result = linear_interpolate_1d_inplace(da, dim="z", where=where)

    assert result is da
    for k in range(n):
        if 1 <= k <= n - 2 and where[k]:
            expected = original[k - 1] + (original[k + 1] - original[k - 1]) / 2
            # atol absorbs subnormal-scale rounding-order differences (numba
            # evaluates the /2 in float64 then stores float32).
            np.testing.assert_allclose(result.values[k], expected, rtol=1e-6, atol=1e-3)
        else:
            # Untouched: endpoints, selected endpoints, and non-selected points.
            np.testing.assert_array_equal(result.values[k], original[k])


@settings(deadline=None, max_examples=200)
@given(data=st.data())
def test_interpolate_recovers_linear_sequence(data: st.DataObject) -> None:
    n = data.draw(_lengths)
    intercept = data.draw(st.floats(-1000, 1000, width=32, allow_nan=False))
    slope = data.draw(st.floats(-100, 100, width=32, allow_nan=False))
    where = _non_consecutive_mask(
        data.draw(st.lists(st.booleans(), min_size=n, max_size=n))
    )

    line = (intercept + slope * np.arange(n)).astype(np.float32)
    values = line.reshape(n, 1, 1).copy()
    values[where, 0, 0] = np.nan  # blank out the points we will interpolate

    da = xr.DataArray(values, dims=["z", "y", "x"])
    result = linear_interpolate_1d_inplace(da, dim="z", where=where)

    # Interior selected points must be recovered to the original line.
    interior = np.zeros(n, dtype=bool)
    interior[1 : n - 1] = where[1 : n - 1]
    np.testing.assert_allclose(
        result.values[interior, 0, 0], line[interior], rtol=1e-4, atol=1e-3
    )


@settings(deadline=None, max_examples=200)
@given(data=st.data())
def test_interpolate_nan_neighbor_propagates(data: st.DataObject) -> None:
    n = data.draw(_lengths)
    values = data.draw(arrays(np.float32, (n, 1, 1), elements=_finite_f32))
    # Pick an interior index to interpolate and force a neighbor to NaN.
    k = data.draw(st.integers(min_value=1, max_value=n - 2))
    nan_side = data.draw(st.sampled_from([-1, 1]))
    values[k + nan_side, 0, 0] = np.nan

    where = np.zeros(n, dtype=bool)
    where[k] = True

    da = xr.DataArray(values.copy(), dims=["z", "y", "x"])
    result = linear_interpolate_1d_inplace(da, dim="z", where=where)
    assert np.isnan(result.values[k, 0, 0])


@settings(deadline=None, max_examples=200)
@given(data=st.data())
def test_interpolate_is_idempotent(data: st.DataObject) -> None:
    n = data.draw(_lengths)
    values = data.draw(arrays(np.float32, (n, 2, 2), elements=_finite_f32))
    where = _non_consecutive_mask(
        data.draw(st.lists(st.booleans(), min_size=n, max_size=n))
    )

    da = xr.DataArray(values.copy(), dims=["z", "y", "x"])
    once = linear_interpolate_1d_inplace(da, dim="z", where=where).values.copy()
    twice = linear_interpolate_1d_inplace(
        xr.DataArray(once.copy(), dims=["z", "y", "x"]), dim="z", where=where
    ).values
    np.testing.assert_array_equal(once, twice)


@settings(deadline=None, max_examples=100)
@given(data=st.data())
def test_interpolate_rejects_consecutive(data: st.DataObject) -> None:
    n = data.draw(st.integers(min_value=4, max_value=20))
    start = data.draw(st.integers(min_value=0, max_value=n - 2))
    where = np.zeros(n, dtype=bool)
    where[start] = where[start + 1] = True  # two consecutive True values

    da = xr.DataArray(np.zeros((n, 1, 1), dtype=np.float32), dims=["z", "y", "x"])
    with pytest.raises(AssertionError, match="consecutive"):
        linear_interpolate_1d_inplace(da, dim="z", where=where)
