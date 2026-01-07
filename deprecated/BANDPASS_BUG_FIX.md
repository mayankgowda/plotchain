# Critical Bandpass Bug Fix

## Bugs Found

### Bug 1: Wrong 3dB Point Calculation Formula

**Problem**: The `_bandpass_3db_points` function used an incorrect formula that didn't actually find the -3dB points.

**Wrong Formula** (old):
```python
# Solving |H|^2 = 1/2, but formula was incorrect
Q2 = Q * Q
a = 2.0 * Q2 + 1.0
disc = max(a * a - 4.0 * Q2, 1e-12)
w2_1 = (a - math.sqrt(disc)) / (2.0 * Q2)
w2_2 = (a + math.sqrt(disc)) / (2.0 * Q2)
w1 = math.sqrt(max(w2_1, 1e-12))
w2 = math.sqrt(max(w2_2, 1e-12))
return float(f0 * w1), float(f0 * w2)
```

**Correct Formula** (fixed):
```python
# Standard formula for 3dB bandwidth of 2nd-order bandpass filter
term = math.sqrt(1.0 + 1.0 / (4.0 * Q * Q))
f1 = f0 * (term - 1.0 / (2.0 * Q))
f2 = f0 * (term + 1.0 / (2.0 * Q))
return float(f1), float(f2)
```

**Impact**: 
- Example: f0=163.4 Hz, Q=2.0
  - **Wrong**: f1=55.9 Hz, f2=238.6 Hz, bandwidth=182.7 Hz
  - **Correct**: f1=127.6 Hz, f2=209.3 Hz, bandwidth=81.7 Hz
  - **Error**: Bandwidth was more than 2x too large!

### Bug 2: Wrong -3dB Horizontal Line Position

**Problem**: The horizontal -3dB line was drawn at an absolute value (-3.0103 dB) instead of relative to the peak.

**Wrong Code** (old):
```python
ax.axhline(-3.0103, linestyle="--", linewidth=1.0, alpha=0.6)
```

**Correct Code** (fixed):
```python
peak_db = float(np.max(mag_db))
ax.axhline(peak_db - 3.0, linestyle="--", linewidth=1.0, alpha=0.6)
```

**Impact**: 
- The horizontal line didn't match where the curve actually crosses -3dB
- Vertical lines (f1, f2) didn't align with the horizontal line
- Models couldn't visually verify the 3dB points

## Verification

### Before Fix (Example: f0=163.4 Hz, Q=2.0)

- Calculated f1=55.9 Hz → magnitude = 0.190370 = **-14.408 dB** ❌ (should be -3dB)
- Calculated f2=238.6 Hz → magnitude = 0.541799 = **-5.323 dB** ❌ (should be -3dB)
- Horizontal line at -3.0103 dB ❌ (doesn't match curve)

### After Fix (Example: f0=163.4 Hz, Q=2.0)

- Calculated f1=127.6 Hz → magnitude = 0.707107 = **-3.010 dB** ✅
- Calculated f2=209.3 Hz → magnitude = 0.707107 = **-3.010 dB** ✅
- Horizontal line at (peak - 3.0) dB ✅ (matches curve)

## Impact on Dataset

**CRITICAL**: All bandpass plots need to be regenerated!

1. **Ground truth values will change**:
   - cp_f1_3db_hz: Will change significantly (e.g., 55.9 → 127.6 Hz)
   - cp_f2_3db_hz: Will change (e.g., 238.6 → 209.3 Hz)
   - bandwidth_hz: Will change significantly (e.g., 182.7 → 81.7 Hz)
   - cp_q_factor: Will change (bandwidth changes)

2. **All evaluations need to be re-run**:
   - Previous model evaluations used wrong ground truth
   - Models may have been reading correctly, but GT was wrong!
   - This explains the high failure rates (80% for cp_f1_3db_hz, 63% for bandwidth_hz)

## Action Required

1. ✅ **Code fixed** - Both bugs corrected in `generate_plotchain_v4.py`
2. ⚠️ **Regenerate dataset** - Run generation script to create new plots
3. ⚠️ **Re-run evaluations** - All model evaluations need to be re-run with correct GT
4. ⚠️ **Update analysis** - Previous failure analysis may be invalid

## Why Models Were Failing

The high failure rates (80% for cp_f1_3db_hz, 63% for bandwidth_hz) were likely because:
1. **Ground truth was wrong** - Models may have been reading correctly, but GT didn't match
2. **Visual mismatch** - Vertical lines didn't align with horizontal -3dB line
3. **Confusing plots** - Models couldn't verify their readings visually

**This bug explains why models struggled with bandpass plots!**

---

**Status**: ✅ Bugs fixed, dataset regeneration required

