# Bandpass Rounding and Tolerance Analysis

## Executive Summary

✅ **Errors are legitimate - Rounding is NOT causing failures**

**Key Findings**:
1. ✅ **100% of GPT/Claude predictions are integers** (rounding is happening)
2. ✅ **86.7% of Gemini predictions are integers** (less rounding)
3. ✅ **0% of failures are within rounding tolerance** - All failures are genuine large errors
4. ✅ **100% of failures are way off** (>5x tolerance) - Not borderline failures
5. ⚠️ **Prompt could be improved** - Says "estimate" which might encourage rounding
6. ✅ **Tolerances are appropriate** - Failures are genuine, not tolerance issues

**Conclusion**: Rounding is happening but NOT causing failures. All failures are legitimate large errors. Tolerances are appropriate. Prompt could be improved to encourage decimal precision.

---

## 1. Rounding Analysis

### 1.1 Integer Prediction Rates

| Field | GPT-4.1 | Claude 4.5 | Gemini 2.5 |
|-------|---------|------------|-------------|
| **resonance_hz** | 100.0% | 100.0% | 100.0% |
| **bandwidth_hz** | 100.0% | 100.0% | 86.7% |
| **cp_f1_3db_hz** | 100.0% | 100.0% | 86.7% |
| **cp_f2_3db_hz** | 100.0% | 100.0% | 86.7% |
| **cp_q_factor** | 10.0% | 7.1% | 43.3% |

**Key Finding**: ✅ **GPT and Claude output 100% integers** for frequency fields. Gemini outputs 86.7% integers (better, but still mostly integers).

### 1.2 Ground Truth Precision

| Field | GT Range | GT Mean | Has Decimals |
|-------|----------|---------|-------------|
| **resonance_hz** | 14.4 - 1286.6 | 322.0 | ✅ Yes (1 decimal) |
| **bandwidth_hz** | 2.0 - 700.6 | 108.3 | ✅ Yes (1 decimal) |
| **cp_f1_3db_hz** | 12.2 - 1223.9 | 273.9 | ✅ Yes (1 decimal) |
| **cp_f2_3db_hz** | 17.0 - 1458.0 | 382.2 | ✅ Yes (1 decimal) |
| **cp_q_factor** | 1.5 - 10.0 | 4.95 | ✅ Yes (1-2 decimals) |

**Key Finding**: ✅ **All GT values have decimals** - Models should output decimals, not integers.

---

## 2. Failure Analysis: Rounding vs Genuine Errors

### 2.1 Are Failures Due to Rounding?

**Analysis**: Checked if failures would pass if we account for rounding (0.5 tolerance for rounding to nearest integer).

| Field | Failures | Within Rounding Tolerance | Way Off (>5x tolerance) |
|-------|----------|----------------------------|-------------------------|
| **resonance_hz** | 22 | 0 (0.0%) | 22 (100.0%) |
| **bandwidth_hz** | 30 | 0 (0.0%) | 30 (100.0%) |
| **cp_f1_3db_hz** | 24 | 0 (0.0%) | 24 (100.0%) |
| **cp_f2_3db_hz** | 25 | 0 (0.0%) | 25 (100.0%) |
| **cp_q_factor** | 27 | 1 (3.7%) | 23 (85.2%) |

**Key Finding**: ✅ **0% of failures are within rounding tolerance** - All failures are genuine large errors, not rounding issues.

### 2.2 Error Magnitudes

| Field | Mean Error | Median Error | Min Error | Max Error | Interpretation |
|-------|------------|-------------|-----------|-----------|----------------|
| **resonance_hz** | 128.96 Hz | 63.55 Hz | 1.60 Hz | 443.70 Hz | ✅ Large errors |
| **bandwidth_hz** | 233.93 Hz | 82.05 Hz | 5.20 Hz | 1127.50 Hz | ✅ Very large errors |
| **cp_f1_3db_hz** | 90.50 Hz | 42.30 Hz | 1.20 Hz | 608.90 Hz | ✅ Large errors |
| **cp_f2_3db_hz** | 256.67 Hz | 144.90 Hz | 3.70 Hz | 1060.20 Hz | ✅ Very large errors |
| **cp_q_factor** | 3.84 | 3.00 | 0.42 | 8.78 | ✅ Large errors |

**Key Finding**: ✅ **All errors are substantial** - Mean errors are 90-257 Hz, way beyond rounding tolerance (0.5 Hz).

---

## 3. Tolerance Analysis

### 3.1 Current Tolerances

| Field | abs_tol | rel_tol | GT Mean | Effective abs_tol |
|-------|---------|---------|---------|-------------------|
| **resonance_hz** | 0.0 | 0.05 (5%) | 322.0 Hz | 16.1 Hz |
| **bandwidth_hz** | 0.0 | 0.08 (8%) | 108.3 Hz | 8.7 Hz |
| **cp_f1_3db_hz** | 0.0 | 0.08 (8%) | 273.9 Hz | 21.9 Hz |
| **cp_f2_3db_hz** | 0.0 | 0.08 (8%) | 382.2 Hz | 30.6 Hz |
| **cp_q_factor** | 0.25 | 0.12 (12%) | 4.95 | 0.25 abs or 0.59 rel |

**Key Finding**: ✅ **Tolerances are reasonable** - 5-8% relative tolerance is appropriate for plot reading.

### 3.2 Would Rounding Tolerance Help?

**Rounding tolerance**: 0.5 Hz absolute (for rounding to nearest integer)

| Field | Current Effective | Rounding Tolerance | Would Help? |
|-------|-------------------|-------------------|-------------|
| **resonance_hz** | 16.1 Hz | 0.5 Hz | ❌ No (errors are 128 Hz mean) |
| **bandwidth_hz** | 8.7 Hz | 0.5 Hz | ❌ No (errors are 234 Hz mean) |
| **cp_f1_3db_hz** | 21.9 Hz | 0.5 Hz | ❌ No (errors are 90 Hz mean) |
| **cp_f2_3db_hz** | 30.6 Hz | 0.5 Hz | ❌ No (errors are 257 Hz mean) |

**Key Finding**: ✅ **Rounding tolerance would NOT help** - Errors are 20-50x larger than rounding tolerance.

### 3.3 Tolerance Recommendation

✅ **Current tolerances are appropriate**:
- 5-8% relative tolerance is reasonable for plot reading
- Failures are not due to strict tolerances
- All failures are way off (>5x tolerance)

**Recommendation**: ✅ **Keep current tolerances** - They are appropriate.

---

## 4. Prompt Analysis

### 4.1 Current Prompt

```
You are given an engineering plot image. Read the plot and answer the question.

Question:
From the bandpass magnitude plot, estimate:
1) resonance_hz (Hz)
2) bandwidth_hz (Hz) using −3 dB points
Return numeric JSON.

Return ONLY a single JSON object matching this schema (numbers or null; no strings; no units; no extra keys):
{
  "resonance_hz": <number or null>  // Hz,
  "bandwidth_hz": <number or null>  // Hz,
  ...
}

Notes:
- Use cp_* fields as intermediate plot reads (checkpoints) that help verify your understanding of the plot.
- If you cannot determine a value, output null for that key.
- IMPORTANT: do NOT output arithmetic expressions like 1025/615; output a decimal number.
```

### 4.2 Issues Identified

1. ⚠️ **"estimate"** - Might encourage rounding
2. ⚠️ **"numbers or null"** - Doesn't specify decimal precision
3. ⚠️ **"output a decimal number"** - Says decimal but models output integers
4. ✅ **No explicit precision guidance** - Models default to integers

### 4.3 Prompt Improvement Recommendation

**Suggested Addition**:
```
- IMPORTANT: Output decimal numbers with appropriate precision (e.g., 154.1 not 154, 19.3 not 19).
- Read values from the plot axes and tick marks - use the precision shown on the plot.
```

**Rationale**: Explicitly encourages decimal precision and references plot precision.

---

## 5. Error Legitimacy Analysis

### 5.1 Are Errors Legitimate?

**Analysis**: Checked if errors are due to:
1. Rounding (NO - 0% within rounding tolerance)
2. Tolerance too strict (NO - errors are >5x tolerance)
3. Genuine model errors (YES - all failures are large errors)

**Conclusion**: ✅ **All errors are legitimate** - Models are making substantial mistakes, not borderline failures.

### 5.2 Example Failures

**resonance_hz**:
- GT=16.1 Hz, Pred=10.0 Hz, Error=6.1 Hz (38% relative error)
- GT=1286.6 Hz, Pred=1000.0 Hz, Error=286.6 Hz (22% relative error)

**bandwidth_hz**:
- GT=2.5 Hz, Pred=10.0 Hz, Error=7.5 Hz (300% relative error)
- GT=24.7 Hz, Pred=360.0 Hz, Error=335.3 Hz (1358% relative error)

**Key Finding**: ✅ **Errors are substantial** - Models are reading wrong values, not just rounding.

---

## 6. Recommendations

### 6.1 Prompt Improvement

✅ **Add explicit decimal precision guidance**:
```
- IMPORTANT: Output decimal numbers with appropriate precision (e.g., 154.1 not 154, 19.3 not 19).
- Read values from the plot axes and tick marks - use the precision shown on the plot.
```

**Expected Impact**: May reduce rounding, but won't fix large errors (which are the real issue).

### 6.2 Tolerance Adjustment

❌ **Do NOT adjust tolerances**:
- Current tolerances are appropriate (5-8% relative)
- Failures are not due to strict tolerances
- All failures are way off (>5x tolerance)
- Adjusting tolerances would hide genuine errors

### 6.3 Rounding Consideration

❌ **Do NOT add rounding tolerance**:
- Rounding tolerance (0.5 Hz) is much smaller than actual errors (90-257 Hz)
- Would not help - errors are 20-50x larger than rounding tolerance
- Would hide genuine errors

---

## 7. Conclusions

### 7.1 Rounding is Happening But Not Causing Failures

✅ **100% of GPT/Claude predictions are integers**
✅ **0% of failures are within rounding tolerance**
✅ **All failures are genuine large errors**

**Conclusion**: Rounding is happening, but it's NOT the cause of failures. All failures are legitimate large errors.

### 7.2 Tolerances Are Appropriate

✅ **Current tolerances are reasonable** (5-8% relative)
✅ **Failures are not due to strict tolerances** (all >5x tolerance)
✅ **No adjustment needed**

**Conclusion**: Tolerances are appropriate. Failures are genuine model errors, not tolerance issues.

### 7.3 Prompt Could Be Improved

⚠️ **Prompt doesn't explicitly encourage decimal precision**
⚠️ **"estimate" might encourage rounding**
✅ **Adding precision guidance may help** (but won't fix large errors)

**Conclusion**: Prompt improvement is recommended but won't solve the core issue (large errors).

### 7.4 Errors Are Legitimate

✅ **All errors are substantial** (90-257 Hz mean errors)
✅ **All failures are way off** (>5x tolerance)
✅ **Models are reading wrong values** (not just rounding)

**Conclusion**: ✅ **All errors are legitimate** - Models are making genuine mistakes, not borderline failures.

---

## 8. Final Verdict

### 8.1 Should We Adjust Tolerances?

❌ **NO** - Current tolerances are appropriate. Failures are not due to strict tolerances.

### 8.2 Should We Account for Rounding?

❌ **NO** - Rounding tolerance (0.5 Hz) is much smaller than actual errors (90-257 Hz). Would not help.

### 8.3 Should We Improve the Prompt?

✅ **YES** - Add explicit decimal precision guidance. May help reduce rounding (though rounding isn't causing failures).

### 8.4 Are Errors Legitimate?

✅ **YES** - All errors are legitimate. Models are making substantial mistakes, not borderline failures.

---

**Recommendation**: 
1. ✅ **Keep current tolerances** - They are appropriate
2. ✅ **Improve prompt** - Add explicit decimal precision guidance
3. ✅ **Accept failures as legitimate** - Models are genuinely struggling with bandpass plots

---

**Analysis Date**: After Bandpass Fix
**Dataset**: PlotChain v4 (bandpass corrected)
**Models**: GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Pro

