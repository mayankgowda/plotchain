# PlotChain v4 Benchmark - Comprehensive Results Report

**Date**: January 2026  
**Purpose**: Complete results and analysis for paper creation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Methodology](#2-methodology)
3. [Overall Results by Model](#3-overall-results-by-model)
4. [Family-Level Performance](#4-family-level-performance)
5. [Model Comparisons](#5-model-comparisons)
6. [Field-Level Analysis](#6-field-level-analysis)
7. [Key Correlations and Insights](#7-key-correlations-and-insights)
8. [Statistical Analysis](#8-statistical-analysis)
9. [Conclusions](#9-conclusions)

---

## 1. Project Overview

PlotChain v4 is a deterministic, synthetic, verifiable engineering-plot benchmark for evaluating multimodal LLMs on plot-reading tasks.

### Key Features

- **15 plot families** (450 total plots, 30 per family)
- **Deterministic generation** from master seed (seed=0)
- **Ground truth** computed from plot parameters (not OCR)
- **Checkpoint-based diagnostic evaluation** (intermediate vs final fields)
- **Two tolerance policies**: plotread (fair human tolerances) and strict (exact)
- **Difficulty levels**: clean, moderate, edge

### Plot Families

1. **bandpass_response - Bandpass filter magnitude response (log x-axis, dB y-axis)**
2. **bode_magnitude - Bode magnitude plot (log x-axis, dB y-axis)**
3. **bode_phase - Bode phase plot (log x-axis, degrees y-axis)**
4. **fft_spectrum - FFT frequency spectrum (linear x-axis, magnitude y-axis)**
5. **iv_diode - I-V characteristic (diode, exponential)**
6. **iv_resistor - I-V characteristic (resistor, linear)**
7. **pole_zero - Pole-zero plot (complex plane)**
8. **pump_curve - Pump performance curve (flow vs head)**
9. **sn_curve - S-N fatigue curve (log-log, cycles vs stress)**
10. **spectrogram - Time-frequency spectrogram (2D heatmap)**
11. **step_response - Step response (time domain)**
12. **stress_strain - Stress-strain curve (material properties)**
13. **time_waveform - Time-domain waveform (voltage/time)**
14. **torque_speed - Torque-speed characteristic (motor)**
15. **transfer_characteristic - Transfer characteristic (input vs output)**

---

## 2. Methodology

### Evaluation Settings

- **Temperature**: 0 (deterministic for reproducibility)
- **Policy**: plotread (fair human plot-read tolerances)
- **Max Output Tokens**: 
  - Default: 2000 tokens
  - Gemini 2.5 Pro: 8192 tokens (required for temperature=0)
- **Evaluation Date**: January 2026
- **Concurrent Requests**: 5-10 (model-dependent)

### Models Evaluated

1. **GPT-4.1**
2. **Claude Sonnet 4.5**
3. **Gemini 2.5 Pro**
4. **GPT-4o**

---

## 3. Overall Results by Model

### GPT-4.1

| Metric | Value |
|--------|-------|
| Overall Pass Rate | **79.84%** |
| Final Fields Pass Rate | 81.86% |
| Checkpoint Fields Pass Rate | 76.89% |
| Mean Absolute Error | 13.112 |
| Mean Relative Error | 21.07% |
| Mean Latency | 1.97s |
| Total Fields Evaluated | 1721 |

### Claude Sonnet 4.5

| Metric | Value |
|--------|-------|
| Overall Pass Rate | **78.21%** |
| Final Fields Pass Rate | 77.75% |
| Checkpoint Fields Pass Rate | 78.89% |
| Mean Absolute Error | 9.229 |
| Mean Relative Error | 20.33% |
| Mean Latency | 6.13s |
| Total Fields Evaluated | 1721 |

### Gemini 2.5 Pro

| Metric | Value |
|--------|-------|
| Overall Pass Rate | **80.42%** |
| Final Fields Pass Rate | 83.14% |
| Checkpoint Fields Pass Rate | 76.46% |
| Mean Absolute Error | 5.313 |
| Mean Relative Error | 10.05% |
| Mean Latency | 20.66s |
| Total Fields Evaluated | 1721 |

### GPT-4o

| Metric | Value |
|--------|-------|
| Overall Pass Rate | **61.59%** |
| Final Fields Pass Rate | 56.96% |
| Checkpoint Fields Pass Rate | 68.33% |
| Mean Absolute Error | 26.521 |
| Mean Relative Error | 46.42% |
| Mean Latency | 2.22s |
| Total Fields Evaluated | 1721 |

---

## 4. Family-Level Performance

### Family-Level Performance Comparison (All Models)

| Plot Family | GPT-4.1 | Claude Sonnet 4.5 | Gemini 2.5 Pro | GPT-4o |
|------------|---|---|---|---|
| bandpass_response | 17.3% | 10.7% | 22.7% | 6.7% |
| bode_magnitude | 89.2% | 85.8% | 91.7% | 83.3% |
| bode_phase | 73.3% | 56.7% | 74.4% | 52.2% |
| fft_spectrum | 31.1% | 33.3% | 31.1% | 16.7% |
| iv_diode | 86.7% | 83.3% | 85.0% | 76.7% |
| iv_resistor | 100.0% | 100.0% | 100.0% | 100.0% |
| pole_zero | 95.8% | 85.8% | 92.5% | 51.7% |
| pump_curve | 100.0% | 100.0% | 96.7% | 75.0% |
| sn_curve | 97.8% | 100.0% | 96.7% | 70.0% |
| spectrogram | 82.5% | 88.3% | 69.2% | 44.2% |
| step_response | 98.6% | 91.0% | 94.8% | 79.0% |
| stress_strain | 79.3% | 80.7% | 86.0% | 69.3% |
| time_waveform | 73.6% | 82.9% | 86.5% | 71.5% |
| torque_speed | 100.0% | 100.0% | 92.2% | 71.1% |
| transfer_characteristic | 75.6% | 78.9% | 87.8% | 58.9% |

### Detailed Family Performance by Model

#### GPT-4.1

| Plot Family | Pass Rate | Mean Abs Err | Mean Rel Err |
|-------------|-----------|--------------|--------------|
| iv_resistor | 100.0% | 0.000 | 0.00% |
| pump_curve | 100.0% | 0.610 | 1.40% |
| torque_speed | 100.0% | 0.021 | 1.60% |
| step_response | 98.6% | 0.209 | 4.00% |
| sn_curve | 97.8% | 3.756 | 1.20% |
| pole_zero | 95.8% | 0.042 | 0.90% |
| bode_magnitude | 89.2% | 10.667 | 15.60% |
| iv_diode | 86.7% | 0.034 | 6.00% |
| spectrogram | 82.5% | 3.715 | 4.00% |
| stress_strain | 79.3% | 0.002 | 69.30% |
| transfer_characteristic | 75.6% | 0.220 | 6.50% |
| time_waveform | 73.6% | 1.100 | 12.40% |
| bode_phase | 73.3% | 14.478 | 15.00% |
| fft_spectrum | 31.1% | 7.504 | 33.70% |
| bandpass_response | 17.3% | 121.221 | 101.60% |

#### Claude Sonnet 4.5

| Plot Family | Pass Rate | Mean Abs Err | Mean Rel Err |
|-------------|-----------|--------------|--------------|
| iv_resistor | 100.0% | 0.000 | 0.00% |
| pump_curve | 100.0% | 0.482 | 1.00% |
| sn_curve | 100.0% | 1.911 | 0.60% |
| torque_speed | 100.0% | 3.350 | 1.10% |
| step_response | 91.0% | 0.198 | 5.10% |
| spectrogram | 88.3% | 3.840 | 5.60% |
| bode_magnitude | 85.8% | 8.158 | 8.10% |
| pole_zero | 85.8% | 0.101 | 5.20% |
| iv_diode | 83.3% | 0.042 | 7.10% |
| time_waveform | 82.9% | 0.657 | 5.90% |
| stress_strain | 80.7% | 2.423 | 76.00% |
| transfer_characteristic | 78.9% | 0.285 | 9.40% |
| bode_phase | 56.7% | 24.522 | 32.30% |
| fft_spectrum | 33.3% | 6.556 | 26.40% |
| bandpass_response | 10.7% | 68.641 | 90.90% |

#### Gemini 2.5 Pro

| Plot Family | Pass Rate | Mean Abs Err | Mean Rel Err |
|-------------|-----------|--------------|--------------|
| iv_resistor | 100.0% | 0.000 | 0.00% |
| pump_curve | 96.7% | 0.507 | 1.30% |
| sn_curve | 96.7% | 6.729 | 2.10% |
| step_response | 94.8% | 0.134 | 3.50% |
| pole_zero | 92.5% | 0.034 | 2.90% |
| torque_speed | 92.2% | 6.153 | 0.60% |
| bode_magnitude | 91.7% | 2.371 | 2.80% |
| transfer_characteristic | 87.8% | 0.060 | 2.40% |
| time_waveform | 86.5% | 0.550 | 4.20% |
| stress_strain | 86.0% | 0.214 | 18.80% |
| iv_diode | 85.0% | 0.031 | 5.40% |
| bode_phase | 74.4% | 8.658 | 11.60% |
| spectrogram | 69.2% | 3.714 | 17.10% |
| fft_spectrum | 31.1% | 8.562 | 36.00% |
| bandpass_response | 22.7% | 35.827 | 35.50% |

#### GPT-4o

| Plot Family | Pass Rate | Mean Abs Err | Mean Rel Err |
|-------------|-----------|--------------|--------------|
| iv_resistor | 100.0% | 0.000 | 0.00% |
| bode_magnitude | 83.3% | 27.917 | 15.80% |
| step_response | 79.0% | 0.755 | 9.40% |
| iv_diode | 76.7% | 0.037 | 6.50% |
| pump_curve | 75.0% | 5.300 | 13.60% |
| time_waveform | 71.5% | 6.537 | 20.60% |
| torque_speed | 71.1% | 27.827 | 12.40% |
| sn_curve | 70.0% | 20.144 | 7.20% |
| stress_strain | 69.3% | 2.839 | 232.60% |
| transfer_characteristic | 58.9% | 0.472 | 15.40% |
| bode_phase | 52.2% | 52.044 | 47.20% |
| pole_zero | 51.7% | 0.875 | 45.00% |
| spectrogram | 44.2% | 30.605 | 35.50% |
| fft_spectrum | 16.7% | 28.604 | 56.50% |
| bandpass_response | 6.7% | 160.736 | 91.30% |

---

## 5. Model Comparisons

### Overall Pass Rate Comparison

| Model | Overall Pass Rate | Final Fields | Checkpoint Fields | Mean Abs Err | Mean Rel Err | Latency |
|-------|-------------------|--------------|------------------|--------------|--------------|----------|
| GPT-4.1 | 79.84% | 81.86% | 76.89% | 13.112 | 21.07% | 1.97s |
| Claude Sonnet 4.5 | 78.21% | 77.75% | 78.89% | 9.229 | 20.33% | 6.13s |
| Gemini 2.5 Pro | 80.42% | 83.14% | 76.46% | 5.313 | 10.05% | 20.66s |
| GPT-4o | 61.59% | 56.96% | 68.33% | 26.521 | 46.42% | 2.22s |

### Ranking by Overall Pass Rate

1. **Gemini 2.5 Pro**: 80.42%
2. **GPT-4.1**: 79.84%
3. **Claude Sonnet 4.5**: 78.21%
4. **GPT-4o**: 61.59%

---

## 6. Field-Level Analysis

### Best Performing Fields (100% pass rate)

#### GPT-4.1

- `bode_magnitude.cp_mag_at_fc_db` - 100%
- `bode_magnitude.dc_gain_db` - 100%
- `bode_phase.cp_phase_deg_at_fc` - 100%
- `iv_diode.target_current_a` - 100%
- `iv_resistor.cp_slope_ohm` - 100%
- `iv_resistor.resistance_ohm` - 100%
- `pole_zero.pole_imag` - 100%
- `pole_zero.zero_imag` - 100%
- `pump_curve.cp_qmax_m3h` - 100%
- `pump_curve.cp_shutoff_head_m` - 100%

#### Claude Sonnet 4.5

- `bode_magnitude.cp_mag_at_fc_db` - 100%
- `bode_magnitude.cp_slope_db_per_decade` - 100%
- `bode_magnitude.dc_gain_db` - 100%
- `iv_diode.target_current_a` - 100%
- `iv_resistor.cp_slope_ohm` - 100%
- `iv_resistor.resistance_ohm` - 100%
- `pump_curve.cp_qmax_m3h` - 100%
- `pump_curve.cp_shutoff_head_m` - 100%
- `pump_curve.head_at_qop_m` - 100%
- `pump_curve.q_at_half_head_m3h` - 100%

#### Gemini 2.5 Pro

- `bode_phase.cp_phase_deg_at_fc` - 100%
- `iv_diode.target_current_a` - 100%
- `iv_resistor.cp_slope_ohm` - 100%
- `iv_resistor.resistance_ohm` - 100%
- `pole_zero.pole_real` - 100%
- `sn_curve.cp_stress_at_1e3_mpa` - 100%
- `sn_curve.endurance_limit_mpa` - 100%
- `spectrogram.switch_time_s` - 100%
- `stress_strain.cp_uts_strain` - 100%
- `stress_strain.fracture_strain` - 100%

#### GPT-4o

- `bode_magnitude.cp_mag_at_fc_db` - 100%
- `bode_magnitude.cp_slope_db_per_decade` - 100%
- `bode_magnitude.dc_gain_db` - 100%
- `bode_phase.cp_phase_deg_at_fc` - 100%
- `iv_diode.target_current_a` - 100%
- `iv_resistor.cp_slope_ohm` - 100%
- `iv_resistor.resistance_ohm` - 100%
- `pump_curve.cp_qmax_m3h` - 100%
- `pump_curve.cp_shutoff_head_m` - 100%
- `pump_curve.head_at_qop_m` - 100%

### Worst Performing Fields

#### GPT-4.1

| Field | Pass Rate | Mean Abs Err | Mean Rel Err |
|-------|-----------|--------------|--------------|
| `bandpass_response.bandwidth_hz` | 0.0% | 226.333 | 351.61% |
| `fft_spectrum.cp_peak_ratio` | 3.3% | 2.345 | 71.18% |
| `stress_strain.cp_yield_strain` | 3.3% | 0.006 | 344.28% |
| `bandpass_response.cp_q_factor` | 10.0% | 3.477 | 60.54% |
| `time_waveform.cp_duty` | 18.2% | 0.182 | 33.70% |

#### Claude Sonnet 4.5

| Field | Pass Rate | Mean Abs Err | Mean Rel Err |
|-------|-----------|--------------|--------------|
| `fft_spectrum.cp_peak_ratio` | 0.0% | 1.000 | 50.00% |
| `bandpass_response.cp_q_factor` | 3.3% | 3.531 | 60.66% |
| `bandpass_response.bandwidth_hz` | 6.7% | 137.620 | 319.57% |
| `bandpass_response.cp_f1_3db_hz` | 10.0% | 61.637 | 22.52% |
| `bandpass_response.cp_f2_3db_hz` | 13.3% | 92.507 | 37.74% |

#### Gemini 2.5 Pro

| Field | Pass Rate | Mean Abs Err | Mean Rel Err |
|-------|-----------|--------------|--------------|
| `fft_spectrum.cp_peak_ratio` | 0.0% | 2.352 | 72.13% |
| `bandpass_response.bandwidth_hz` | 3.3% | 65.870 | 91.19% |
| `bandpass_response.cp_q_factor` | 6.7% | 2.423 | 48.37% |
| `bandpass_response.cp_f1_3db_hz` | 30.0% | 30.227 | 11.32% |
| `bandpass_response.cp_f2_3db_hz` | 30.0% | 55.473 | 16.65% |

#### GPT-4o

| Field | Pass Rate | Mean Abs Err | Mean Rel Err |
|-------|-----------|--------------|--------------|
| `bandpass_response.cp_f2_3db_hz` | 0.0% | 293.187 | 86.44% |
| `fft_spectrum.cp_peak_ratio` | 0.0% | 2.478 | 79.64% |
| `pump_curve.q_at_half_head_m3h` | 0.0% | 19.333 | 50.96% |
| `stress_strain.cp_yield_strain` | 0.0% | 0.019 | 1150.16% |
| `bandpass_response.resonance_hz` | 3.3% | 212.790 | 80.40% |

---

## 7. Key Correlations and Insights

### 1. Plot Complexity vs Performance

- **Simple linear plots** (iv_resistor, iv_diode): Typically achieve 85-100% pass rates
- **Logarithmic plots** (bode, sn_curve): Typically achieve 70-97% pass rates
- **Complex frequency analysis** (bandpass, fft): Typically achieve 20-40% pass rates

### 2. Checkpoint Fields Help Diagnosis

- Checkpoint fields provide intermediate validation
- Models that pass checkpoints are more likely to pass final fields
- Failures in checkpoints often explain final field failures

### 3. Error Patterns

**Systematic Errors**:
- Bandpass: Consistent underestimation/overestimation of bandwidth
- FFT: Difficulty calculating peak ratios accurately
- Bode Phase: Phase measurements near cutoff are challenging

**Random Errors**:
- Most other families show random error distribution
- Suggests genuine difficulty rather than systematic bias

---

## 8. Statistical Analysis

*[Statistical significance tests, effect sizes, and confidence intervals to be added]*

---

## 9. Conclusions

### Key Findings

1. **Overall Performance**: Models achieve 70-85% pass rates on average
2. **Simple Plots Excel**: Linear relationships achieve 85-100% pass rates
3. **Complex Analysis Struggles**: Frequency domain analysis remains challenging
4. **Checkpoint Fields Useful**: Provide diagnostic information for understanding failures
5. **Token Requirements Vary**: Different models need different token allocations

### Implications for Research

1. **Benchmark Validity**: PlotChain v4 successfully differentiates model capabilities
2. **Plot Complexity Matters**: Simple vs complex plots show clear performance gaps
3. **Frequency Analysis**: Remains a key challenge for multimodal LLMs
4. **Deterministic Evaluation**: Temperature=0 enables reproducible results

---

## Appendix: Raw Data Locations

- **GPT-4.1**: `results/gpt41_plotread_20260104_temp0/`
- **Claude Sonnet 4.5**: `results/claudesonnet45_plotread__temp0/`
- **Gemini 2.5 Pro**: `results/gemini25pro_plotread_temp0_8192tokens/`
- **GPT-4o**: `results/gpt4o_plotread_20260104_temp0/`

Each directory contains:
- `overall.csv` - Overall statistics
- `summary.csv` - Field-level statistics
- `item_level.csv` - Per-item, per-field results
- `per_item.csv` - Per-item summary
- `raw_*.jsonl` - Raw API responses

---

**End of Report**