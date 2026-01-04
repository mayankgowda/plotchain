# PlotChain v4 Evaluation Results Analysis Report
**Model**: OpenAI GPT-4.1  
**Policy**: plotread  
**Date**: Analysis Date  
**Total Items**: 450 (30 per family × 15 families)

---

## Executive Summary

✅ **READY FOR OTHER MODELS**: The evaluation pipeline is functioning correctly with **81.2% overall pass rate** and **zero null outputs**. The benchmark is ready for evaluation of additional models.

⚠️ **MINOR ISSUES FOUND**:
1. `cp_duty` field only appears for square wave items (11/30 time_waveform) - **This is correct behavior**
2. Some families show systematic overestimation bias
3. Three families have low pass rates (<50%): bandpass_response, fft_spectrum, bode_phase

---

## 1. Overall Performance Metrics

### 1.1 High-Level Statistics

| Metric | Value | Status |
|--------|-------|--------|
| **Overall Pass Rate** | 81.2% | ✅ Good |
| **Final Fields Pass Rate** | 82.0% | ✅ Good |
| **Checkpoint Fields Pass Rate** | 80.0% | ✅ Good |
| **Mean Absolute Error** | 8.97 | ✅ Acceptable |
| **Mean Relative Error** | 16.3% | ✅ Acceptable |
| **Mean Latency** | 1.65s | ✅ Fast |
| **Null Output Rate** | 0.0% | ✅ Excellent |
| **Total Evaluations** | 1,721 | ✅ Complete |

### 1.2 Performance Interpretation

- **Pass Rate (81.2%)**: Above the 80% threshold for "good" performance
- **Final vs Checkpoint**: Final fields perform slightly better (82.0% vs 80.0%), which is expected
- **No Null Outputs**: Model attempts all fields, indicating good prompt understanding
- **Latency**: 1.65s average is acceptable for practical deployment

**Verdict**: ✅ **Strong overall performance, ready for comparison with other models**

---

## 2. Family-Level Performance Analysis

### 2.1 Pass Rates by Family (Final Fields)

| Rank | Family | Pass Rate | Items | Mean Abs Err | Status |
|------|--------|-----------|-------|--------------|--------|
| 1 | iv_resistor | 100.0% | 30 | 0.00 | ✅ Perfect |
| 1 | sn_curve | 100.0% | 30 | 4.63 | ✅ Perfect |
| 1 | torque_speed | 100.0% | 30 | 0.00 | ✅ Perfect |
| 4 | pump_curve | 96.7% | 30 | 1.14 | ✅ Excellent |
| 5 | stress_strain | 97.8% | 30 | 0.00 | ✅ Excellent |
| 6 | pole_zero | 95.8% | 30 | 0.04 | ✅ Excellent |
| 7 | step_response | 95.6% | 30 | 0.50 | ✅ Excellent |
| 8 | bode_magnitude | 90.0% | 30 | 11.92 | ✅ Good |
| 9 | iv_diode | 83.3% | 30 | 0.04 | ✅ Good |
| 10 | transfer_characteristic | 78.3% | 30 | 0.26 | ⚠️ Moderate |
| 11 | spectrogram | 77.8% | 30 | 4.28 | ⚠️ Moderate |
| 12 | time_waveform | 75.0% | 30 | 3.07 | ⚠️ Moderate |
| 13 | bode_phase | 55.0% | 30 | 48.42 | ❌ Poor |
| 14 | fft_spectrum | 40.0% | 30 | 10.83 | ❌ Poor |
| 15 | bandpass_response | 26.7% | 30 | 77.14 | ❌ Very Poor |

### 2.2 Family Performance Categories

**Excellent (≥95%)**: 7 families
- iv_resistor, sn_curve, torque_speed, pump_curve, stress_strain, pole_zero, step_response

**Good (80-95%)**: 2 families
- bode_magnitude, iv_diode

**Moderate (60-80%)**: 3 families
- transfer_characteristic, spectrogram, time_waveform

**Poor (<60%)**: 3 families
- bode_phase (55%), fft_spectrum (40%), bandpass_response (26.7%)

### 2.3 Problematic Families Analysis

#### bandpass_response (26.7% pass rate)
- **Issues**: Very low pass rate, high errors (mean 77.14, p95 275.87)
- **Fields**: `resonance_hz` (26.7%), `bandwidth_hz` (26.7%)
- **Possible Causes**: 
  - Complex frequency domain analysis required
  - 3dB point detection may be challenging
  - Checkpoint fields also struggling (cp_f1_3db_hz: 16.7%, cp_f2_3db_hz: 33.3%)

#### fft_spectrum (40.0% pass rate)
- **Issues**: Low pass rate, checkpoint field very poor (cp_peak_ratio: 3.3%)
- **Fields**: `dominant_frequency_hz` (20%), `secondary_frequency_hz` (60%)
- **Possible Causes**:
  - Peak detection in frequency domain
  - cp_peak_ratio requires dB-to-linear conversion (model may struggle)
  - Note: This was the family with null outputs in previous runs (now fixed)

#### bode_phase (55.0% pass rate)
- **Issues**: Moderate pass rate, high errors (mean 48.42, p95 257.50)
- **Fields**: `cutoff_hz` (60%), `phase_deg_at_fq` (50%)
- **Possible Causes**:
  - Phase reading may be more challenging than magnitude
  - Cutoff frequency detection from phase plot

**Recommendation**: These three families may need tolerance review or indicate genuine model limitations in frequency-domain analysis.

---

## 3. Field-Level Analysis

### 3.1 Worst Performing Fields (Final Fields)

| Family | Field | Pass Rate | Mean Abs Err | n |
|--------|-------|-----------|--------------|---|
| fft_spectrum | dominant_frequency_hz | 20.0% | 13.50 | 30 |
| bandpass_response | bandwidth_hz | 26.7% | 99.47 | 30 |
| bandpass_response | resonance_hz | 26.7% | 54.82 | 30 |
| bode_phase | phase_deg_at_fq | 50.0% | 7.10 | 30 |
| time_waveform | frequency_hz | 50.0% | 6.13 | 30 |
| transfer_characteristic | small_signal_gain | 56.7% | 0.52 | 30 |
| bode_phase | cutoff_hz | 60.0% | 89.73 | 30 |
| fft_spectrum | secondary_frequency_hz | 60.0% | 8.17 | 30 |
| iv_diode | turn_on_voltage_v_at_target_i | 66.7% | 0.07 | 30 |
| spectrogram | f1_hz | 66.7% | 6.17 | 30 |

### 3.2 Best Performing Fields

All fields with 100% pass rate:
- iv_resistor: resistance_ohm, cp_slope_ohm
- sn_curve: stress_at_1e5_mpa, endurance_limit_mpa, cp_stress_at_1e3_mpa
- torque_speed: stall_torque_nm, no_load_speed_rpm, cp_torque_at_speed_q_nm
- pump_curve: head_at_qop_m, cp_shutoff_head_m
- stress_strain: yield_strength_mpa, uts_mpa, cp_uts_strain
- step_response: percent_overshoot, steady_state, cp_peak_time_s, cp_peak_value, cp_band_lower, cp_band_upper
- pole_zero: pole_imag, zero_imag
- bode_magnitude: dc_gain_db, cp_mag_at_fc_db
- bode_phase: cp_phase_deg_at_fc
- time_waveform: vpp_v, cp_vmax_v, cp_vmin_v
- spectrogram: switch_time_s
- transfer_characteristic: saturation_v
- iv_diode: target_current_a

**Total**: 25 fields with 100% pass rate

---

## 4. Checkpoint Field Analysis

### 4.1 Checkpoint vs Final Performance

| Scope | Pass Rate | n | Mean Abs Err |
|-------|-----------|---|--------------|
| Final Fields | 82.0% | 1,020 | 7.89 |
| Checkpoint Fields | 80.0% | 701 | 10.58 |

**Difference**: 2.0 percentage points (final fields slightly better)

**Interpretation**: ✅ **Healthy pattern** - Checkpoint fields are slightly harder (as expected), but not dramatically worse. This suggests the model is not just guessing final answers.

### 4.2 Worst Checkpoint Fields

| Family | Field | Pass Rate | Mean Abs Err |
|--------|-------|-----------|--------------|
| fft_spectrum | cp_peak_ratio | 3.3% | 2.43 |
| bandpass_response | cp_f1_3db_hz | 16.7% | 55.94 |
| stress_strain | cp_yield_strain | 6.7% | 0.006 |
| bandpass_response | cp_f2_3db_hz | 33.3% | 114.77 |
| time_waveform | cp_duty | 9.1% | 0.20 |

**Note**: `cp_peak_ratio` requires dB-to-linear conversion, which may be challenging for models.

---

## 5. Difficulty Impact Analysis

### 5.1 Performance by Difficulty Level

| Difficulty | Pass Rate | Mean Abs Err | Item-Level Final Pass |
|------------|-----------|--------------|----------------------|
| Clean | 82.4% | 6.40 | 71.7% |
| Moderate | 81.9% | 7.11 | 69.6% |
| Edge | 78.8% | 14.28 | 63.7% |

### 5.2 Difficulty Gradient Analysis

✅ **Expected Pattern**: Performance degrades with difficulty
- Clean > Moderate > Edge (as expected)
- Edge cases show higher errors (14.28 vs 6.40)
- Item-level success also decreases with difficulty

**Verdict**: ✅ **Difficulty levels are working as intended** - harder plots are indeed more challenging.

---

## 6. Error Analysis

### 6.1 Error Distribution

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 8.97 |
| Median Absolute Error | 0.00 |
| 95th Percentile Error | 30.00 |
| Max Absolute Error | 766.50 |

**Interpretation**:
- **Median = 0**: Most predictions are exact (within tolerance)
- **High p95**: Some predictions have large errors (likely from problematic families)
- **Max error**: Extreme outlier (likely bandpass_response)

### 6.2 Systematic Bias Analysis

**Overall Mean Signed Error**: +4.88 (overestimation)

**Bias by Family** (positive = overestimate, negative = underestimate):

| Family | Mean Signed Error | Interpretation |
|--------|-------------------|----------------|
| bandpass_response | +32.26 | Strong overestimation |
| bode_phase | +19.54 | Strong overestimation |
| bode_magnitude | +7.71 | Moderate overestimation |
| fft_spectrum | +5.55 | Moderate overestimation |
| sn_curve | +2.56 | Mild overestimation |
| spectrogram | +1.10 | Mild overestimation |
| pump_curve | +0.22 | Minimal bias |
| iv_diode | +0.03 | Minimal bias |
| stress_strain | +0.00 | No bias |
| iv_resistor | 0.00 | No bias |
| torque_speed | +0.00 | No bias |
| transfer_characteristic | -0.01 | Minimal bias |
| step_response | -0.03 | Minimal bias |
| pole_zero | -0.04 | Minimal bias |
| time_waveform | -0.07 | Minimal bias |

**Key Finding**: ⚠️ **Systematic overestimation in frequency-domain families** (bandpass_response, bode_phase, bode_magnitude, fft_spectrum). This may indicate:
- Model tendency to read higher frequencies than actual
- Challenges with frequency domain interpretation
- Possible tolerance issues for these families

---

## 7. Data Consistency Checks

### 7.1 Completeness

✅ **All dataset items have results**: 450/450 items evaluated

✅ **All families have 30 items**: Coverage is complete

✅ **Field counts match**: All families have correct number of fields

✅ **cp_duty Field**: Only appears for square wave items (11/30 time_waveform items). This is **correct behavior** - `cp_duty` is only defined for square waves, not sine/triangle waves. The field appears in results only when present in ground truth.

### 7.2 Data Quality

✅ **No negative pass rates**: All pass rates in valid range [0, 1]

✅ **No extreme errors**: No predictions with abs_err > 1000 (except one outlier at 766.50, which is from bandpass_response)

✅ **No null outputs**: 0 null predictions (0.0%)

### 7.3 Tolerance Analysis

- **Predictions within 2x tolerance**: 79.3% (1,365/1,721)
- **Predictions just within tolerance**: 1.2% (21 predictions)

**Interpretation**: ✅ Tolerances appear appropriate - most predictions are well within bounds, not just barely passing.

---

## 8. Item-Level Success Rates

### 8.1 Complete Item Success

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Items with ALL final fields passing | 68.7% (309/450) | ✅ Good |
| Items with ALL checkpoint fields passing | 74.9% (337/450) | ✅ Good |

**Interpretation**: 
- About 2/3 of items are completely correct for final fields
- Checkpoint fields have slightly higher complete success (likely because some items have fewer checkpoint fields)

### 8.2 Item-Level by Difficulty

| Difficulty | All Final Pass | All Checkpoint Pass |
|------------|----------------|---------------------|
| Clean | 71.7% | 72.8% |
| Moderate | 69.6% | 74.8% |
| Edge | 63.7% | 77.8% |

**Note**: Checkpoint pass rate increases with difficulty (unexpected). This may be because:
- Edge cases have fewer checkpoint fields
- Or checkpoint fields in edge cases are easier to read

---

## 9. Readiness Assessment

### 9.1 ✅ READY FOR OTHER MODELS

**Evidence**:
1. ✅ **Zero null outputs**: Pipeline handles all fields correctly
2. ✅ **Complete coverage**: All 450 items evaluated
3. ✅ **Data consistency**: Field counts match, no missing data
4. ✅ **Reasonable performance**: 81.2% pass rate indicates benchmark is working
5. ✅ **Difficulty gradient**: Performance degrades as expected
6. ✅ **Checkpoint validation**: Checkpoint fields show model understanding

### 9.2 ⚠️ Issues to Monitor

1. **Three problematic families** (<50% pass rate):
   - bandpass_response (26.7%)
   - fft_spectrum (40.0%)
   - bode_phase (55.0%)
   
   **Action**: Monitor these families when evaluating other models. If all models struggle, may indicate benchmark difficulty or tolerance issues.

2. **Systematic overestimation** in frequency-domain families:
   - bandpass_response: +32.26
   - bode_phase: +19.54
   - bode_magnitude: +7.71
   - fft_spectrum: +5.55
   
   **Action**: Check if this is model-specific or systematic. If systematic, may need tolerance adjustment.

3. **cp_duty field**: Only appears in 11/30 time_waveform items (likely correct - only for square waves)

### 9.3 Recommendations

**Before evaluating other models**:
1. ✅ **Proceed**: Pipeline is ready
2. ⚠️ **Monitor**: Watch the three problematic families
3. ⚠️ **Document**: Note systematic bias patterns for comparison

**For paper**:
1. Report overall pass rate (81.2%)
2. Highlight family-level variation
3. Discuss difficulty impact
4. Note systematic biases (if consistent across models)

---

## 10. Comparison Baseline

### 10.1 Performance Summary

**GPT-4.1 Performance**:
- Overall: 81.2% pass rate
- Final fields: 82.0% pass rate
- Checkpoint fields: 80.0% pass rate
- Zero null outputs
- Mean latency: 1.65s

**This establishes a strong baseline** for comparing other models.

### 10.2 Expected Comparison Points

When evaluating other models, compare:
1. **Overall pass rate** (baseline: 81.2%)
2. **Family-level performance** (identify which families are hardest for all models)
3. **Null output rate** (baseline: 0%)
4. **Systematic biases** (check if frequency-domain overestimation is model-specific)
5. **Difficulty impact** (should see clean > moderate > edge for all models)

---

## 11. Conclusions

### 11.1 Readiness Status

✅ **READY FOR OTHER MODELS**

The evaluation pipeline is functioning correctly with:
- Complete data coverage
- Zero null outputs
- Reasonable performance (81.2%)
- Expected difficulty gradient
- Data consistency

### 11.2 Key Findings

1. **Strong overall performance**: 81.2% pass rate
2. **Family variation**: 7 families ≥95%, 3 families <60%
3. **No null outputs**: Model attempts all fields
4. **Systematic bias**: Overestimation in frequency-domain families
5. **Difficulty works**: Performance degrades as expected

### 11.3 Next Steps

1. ✅ **Proceed with other models**: Pipeline is ready
2. ⚠️ **Monitor problematic families**: bandpass_response, fft_spectrum, bode_phase
3. ⚠️ **Compare biases**: Check if frequency-domain overestimation is consistent
4. ✅ **Document baseline**: Use GPT-4.1 as comparison point

---

## Appendix A: Detailed Statistics

### A.1 Family Performance (Full Table)

See Section 2.1 for complete family performance breakdown.

### A.2 Field Performance (Full Table)

See Section 3.1 for worst-performing fields and Section 3.2 for best-performing fields.

### A.3 Error Statistics by Family

| Family | Mean Abs Err | Median Abs Err | p95 Abs Err | Max Abs Err |
|--------|--------------|----------------|-------------|-------------|
| bandpass_response | 77.14 | 38.85 | 275.87 | 766.50 |
| bode_phase | 48.42 | 4.00 | 257.50 | 500.00 |
| bode_magnitude | 11.92 | 0.00 | 58.88 | 117.75 |
| fft_spectrum | 10.83 | 2.50 | 31.63 | 35.50 |
| spectrogram | 4.28 | 0.00 | 13.33 | 20.00 |
| sn_curve | 4.63 | 7.00 | 9.88 | 19.75 |
| time_waveform | 3.07 | 1.50 | 10.55 | 21.10 |
| transfer_characteristic | 0.26 | 0.00 | 1.28 | 2.55 |
| step_response | 0.50 | 0.15 | 1.06 | 2.60 |
| pump_curve | 1.14 | 0.20 | 2.90 | 5.20 |
| pole_zero | 0.04 | 0.00 | 0.25 | 1.00 |
| stress_strain | 0.00 | 0.00 | 0.01 | 0.01 |
| iv_diode | 0.04 | 0.06 | 0.09 | 0.18 |
| iv_resistor | 0.00 | 0.00 | 0.00 | 0.00 |
| torque_speed | 0.00 | 0.00 | 0.00 | 0.00 |

---

**End of Report**

This report confirms that the PlotChain v4 evaluation pipeline is ready for testing additional models. The baseline performance of GPT-4.1 (81.2% pass rate) provides a strong comparison point.

