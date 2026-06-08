# Property-based tests for numerical/numba functions

*2026-06-06T12:09:05Z by Showboat 0.6.1*
<!-- showboat-id: c0f03234-e94d-4a2a-83f7-3a483604d45f -->

## Objective

Read the Hypothesis docs, examine the project's numba / numerical functions,
identify their key invariants and testing gaps, and backfill **property-based
tests** with Hypothesis that verify and demonstrate correctness — fixing any
issues the new tests surface.

## Functions in scope

A repo-wide search for `@njit` and other numerical helpers finds exactly three
numba-accelerated numerical kernels, all in `src/reformatters/common/`:

1. `binary_rounding.round_float32_inplace` — IEEE-754 round-half-to-even
   truncation of float32 mantissa bits (improves compression).
2. `interpolation.linear_interpolate_1d_inplace` — midpoint interpolation of
   isolated gaps along an axis.
3. `deaccumulation.deaccumulate_to_rates_inplace` — converts accumulated /
   running-mean source values into per-second rates.

These power data quality and storage for every weather dataset, so silent
numerical errors would corrupt published archives.

## Invariants identified (and the testing gaps)

The repo had only example-based unit tests. The property-based suite adds the
following invariants:

**round_float32_inplace**
- Matches an independent round-half-to-even reference computed with exact
  rational arithmetic (`fractions.Fraction`) — not the bit logic under test.
- Special values (+/-0, +/-inf, NaN) preserved exactly.  *(gap -> bug, below)*
- Idempotent; order-preserving (monotonic); sign-symmetric `round(-x)==-round(x)`.
- Finite results have zeroed dropped mantissa bits; relative error <= 2**-keep.

**linear_interpolate_1d_inplace**
- Interpolated points equal the midpoint of their two original neighbors.
- Non-selected points and endpoints unchanged; NaN neighbor propagates.
- A linear sequence is recovered exactly; idempotent; rejects consecutive gaps.

**deaccumulate_to_rates_inplace**
- Round-trip: integrating per-step rates and deaccumulating recovers the rates.
  Exact for integer "accumulated" inputs; within tolerance for "running_mean".
- First step always NaN; non-negative input never clamps/NaNs; holds across
  parallelised leading/trailing dimensions.

## Bug found: round_float32_inplace turns NaN into inf

The "special values preserved" property failed. With a small `keep_mantissa_bits`
the rounding logic rounds the NaN's mantissa **up**, it overflows, and the carry
propagates into NaN's all-ones exponent field — producing **inf**.

A standard quiet NaN (0x7FC00000) survives the keep=7/8/10 values used in
production today, so live archives are not currently corrupted. But this is a
latent correctness bug: the function is documented and unit-tested to preserve
special values, yet at low `keep` (or other NaN payloads from upstream ops) it
silently converts missing-data NaN into inf — which is far more damaging in a
weather archive (it poisons min/max/mean and plotting).

The demo below loads the **committed (pre-fix)** function straight from git and
contrasts it with the fixed version.

```bash
.venv/bin/python - <<'PY'
import subprocess, types
import numpy as np
src = subprocess.run(
    ["git", "show", "HEAD:src/reformatters/common/binary_rounding.py"],
    capture_output=True, text=True, check=True,
).stdout
old = types.ModuleType("old_binary_rounding")
exec(compile(src, "old_binary_rounding.py", "exec"), old.__dict__)
from reformatters.common.binary_rounding import round_float32_inplace as fixed
mk = lambda: np.array([np.nan], dtype=np.float32)
print("keep_mantissa_bits=0, input = NaN")
print("  committed (pre-fix):", old.round_float32_inplace(mk(), 0)[0])
print("  with fix          :", fixed(mk(), 0)[0])
PY
```

```output
keep_mantissa_bits=0, input = NaN
  committed (pre-fix): inf
  with fix          : nan
```

## The fix

Skip non-finite values in the kernel: their exponent is already all-ones, so
leaving them untouched is the correct (identity) rounding behavior for inf/NaN
and removes any chance of a mantissa carry corrupting the exponent.

```bash
git -c color.ui=never diff src/reformatters/common/binary_rounding.py
```

```output
diff --git a/src/reformatters/common/binary_rounding.py b/src/reformatters/common/binary_rounding.py
index de100d0..52fe21a 100644
--- a/src/reformatters/common/binary_rounding.py
+++ b/src/reformatters/common/binary_rounding.py
@@ -56,6 +56,12 @@ def _round_float32_inplace_numba(
     flat_bits = bits.ravel()  # modify 1D view in place
 
     for i in prange(len(flat_bits)):  # ty: ignore[not-iterable]
+        # Leave non-finite values (inf, nan) unchanged. Their exponent is all
+        # ones; rounding their mantissa up can carry into that exponent and
+        # silently turn a nan into an inf, corrupting missing-data sentinels.
+        if (flat_bits[i] & exponent_mask) == exponent_mask:
+            continue
+
         mantissa = flat_bits[i] & mantissa_mask
         round_bit = flat_bits[i] & np.uint32(1 << drop_bits)
         half_bit = flat_bits[i] & np.uint32(1 << (drop_bits - 1))
```

## Verification

`hypothesis` was added as a dev dependency. The property tests live alongside
the existing unit tests in `tests/common/`.

Environment caveat: this container runs Python 3.14.0rc2 + pydantic 2.13.4,
where pydantic's model construction calls `typing._eval_type(..., prefer_fwd_module=True)`,
a kwarg that does not exist in 3.14.0rc2. That breaks `tests/conftest.py` import
(and thus full pytest collection) repo-wide — unrelated to these changes. The
numerical tests need no pydantic, so they are run with `--noconftest`. Under the
project's normal Python they run as ordinary pytest tests.

```bash
.venv/bin/python -m pytest tests/common/test_binary_rounding.py --noconftest -o addopts='' -p no:cacheprovider -q -p no:sugar 2>&1 | tail -3 | sed -E 's/ in [0-9.]+s$//'
```

```output
................................                                         [100%]
32 passed
```

```bash
.venv/bin/python -m pytest tests/common/test_interpolation.py --noconftest -o addopts='' -p no:cacheprovider -q -p no:sugar 2>&1 | tail -3 | sed -E 's/ in [0-9.]+s$//'
```

```output
........                                                                 [100%]
8 passed
```

```bash
.venv/bin/python -m pytest tests/common/test_deaccumulation.py --noconftest -o addopts='' -p no:cacheprovider -q -p no:sugar 2>&1 | tail -3 | sed -E 's/ in [0-9.]+s$//'
```

```output
....................................                                     [100%]
36 passed
```

## Summary

- Added Hypothesis property-based tests for all three numerical numba kernels
  (7 binary-rounding, 5 interpolation, 4 deaccumulation properties), each
  encoding the invariants listed above. 76 tests pass (existing + new).
- Found and fixed a real correctness bug: `round_float32_inplace` could turn
  NaN into inf. The behavior is now pinned by `test_round_preserves_special_values`.
- Two initial property failures were defects in the *test references* (a
  subnormal rounding-order tolerance and an off-by-one in the running-mean
  energy integral), corrected without weakening the invariants.
- Added `hypothesis` to the dev dependency group.

