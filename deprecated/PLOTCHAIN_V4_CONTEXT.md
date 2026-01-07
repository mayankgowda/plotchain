# PlotChain v4: Engineering Plot Reading Benchmark - Complete Context Document

## Executive Summary

PlotChain v4 is a **synthetic, deterministic, verifiable benchmark** for evaluating multimodal LLMs' ability to read engineering plots and extract quantitative information. The benchmark consists of **15 plot families** with **30 items each** (450 total items), covering diverse engineering domains including control systems, signal processing, materials science, and mechanical engineering.

**Key Innovation**: Unlike OCR-based benchmarks, PlotChain uses **deterministic ground truth** computed from plot generation parameters, ensuring 100% accurate evaluation without human annotation errors.

---

## 1. Dataset Overview

### 1.1 Dataset Structure

**Location**: `data/plotchain_v4/`

**Files**:
- `plotchain_v4.jsonl` - One JSON record per item (450 items)
- `images/<family>/<family>_<idx>.png` - Plot images (450 images)
- `validation_rows.csv` - Baseline validation results
- `validation_summary.csv` - Validation summary by type/field

### 1.2 JSONL Record Structure

Each item in `plotchain_v4.jsonl` contains:

```json
{
  "id": "step_response_000",
  "type": "step_response",
  "image_path": "images/step_response/step_response_000.png",
  "question": "From the step response plot, estimate:\n1) percent_overshoot (%)\n2) settling_time_s (2% criterion, seconds)\n3) steady_state (final value)\nReturn numeric JSON.",
  "ground_truth": {
    "percent_overshoot": 52.4,
    "settling_time_s": 4.95,
    "steady_state": 1.0,
    "cp_peak_time_s": 0.8,
    "cp_peak_value": 1.53,
    "cp_band_lower": 0.98,
    "cp_band_upper": 1.02
  },
  "plot_params": {
    "zeta": 0.2,
    "wn_rad_s": 4.0,
    "t_end_s": 7.5,
    "n_points": 700
  },
  "generation": {
    "seed": 1488399605,
    "difficulty": "clean",
    "edge_tag": "",
    "final_fields": ["percent_overshoot", "settling_time_s", "steady_state"],
    "checkpoint_fields": ["cp_peak_time_s", "cp_peak_value", "cp_band_lower", "cp_band_upper"],
    "x_min": 0.0,
    "x_max": 7.5,
    "y_min": 0.0,
    "y_max": 1.53,
    "tick_step_x": 2.0,
    "tick_step_y": 0.5
  }
}
```

**Key Fields**:
- `id`: Unique identifier (`<family>_<index>`)
- `type`: Plot family name
- `image_path`: Relative path to PNG image
- `question`: Text question asking for specific values
- `ground_truth`: Dictionary of expected values (gold standard)
- `plot_params`: Parameters used to generate the plot
- `generation.final_fields`: Fields that are the primary answers
- `generation.checkpoint_fields`: Intermediate fields (cp_*) for debugging

### 1.3 Plot Families (15 total)

1. **step_response** - 2nd-order system step response
2. **bode_magnitude** - Bode magnitude plot (1st-order low-pass)
3. **bode_phase** - Bode phase plot (1st-order low-pass)
4. **bandpass_response** - Bandpass filter magnitude response
5. **time_waveform** - Time-domain waveforms (sine/square/triangle)
6. **fft_spectrum** - FFT magnitude spectrum (two-tone)
7. **spectrogram** - Spectrogram (tone switch)
8. **iv_resistor** - IV curve for resistor (V vs I)
9. **iv_diode** - IV curve for diode (I vs V)
10. **transfer_characteristic** - Transfer characteristic (saturating amplifier)
11. **pole_zero** - Pole-zero plot (single pole + zero)
12. **stress_strain** - Stress-strain curve (materials)
13. **torque_speed** - Torque-speed curve (motors)
14. **pump_curve** - Pump curve (head vs flow)
15. **sn_curve** - S-N curve (fatigue)

### 1.4 Difficulty Levels

Each family has a **40/30/30 split**:
- **40% clean**: Full grid, markers, high resolution, no noise
- **30% moderate**: Grid, some noise, moderate resolution
- **30% edge**: No grid, more noise, coarse ticks, challenging conditions

---

## 2. Ground Truth Generation

### 2.1 Deterministic Computation

**Critical Feature**: Ground truth is **NOT** from OCR or human annotation. Instead:

1. **Plot Parameters** (`plot_params`) define the mathematical model
2. **Baseline Functions** compute exact values from parameters
3. **Quantization** rounds to human-friendly precision (via `ANSWER_FORMAT`)
4. **Validation** verifies baseline matches stored GT (100% pass rate expected)

**Example**: For step response:
- Parameters: `zeta=0.2, wn_rad_s=4.0`
- Baseline computes: `percent_overshoot`, `settling_time_s`, `steady_state`
- Values quantized to specified decimal places

### 2.2 Field Types

**Final Fields**: Primary answers requested in the question
- Example: `percent_overshoot`, `settling_time_s`, `steady_state`

**Checkpoint Fields** (`cp_*`): Intermediate plot reads for verification
- Example: `cp_peak_time_s`, `cp_peak_value`, `cp_band_lower`
- Help verify model can read intermediate values
- Not always explicitly asked in question, but included in schema

### 2.3 Human-Friendly Design

- **Tick-aligned values**: Avoids weird fractions
- **Proper precision**: Each field has specified decimal places
- **Realistic ranges**: Engineering-appropriate values
- **Good variation**: ~80-90% unique combinations per family

---

## 3. Evaluation Methodology

### 3.1 Evaluation Script

**File**: `run_plotchain_v4_eval.py`

**Modes**:
- `--mode run`: Call models via API, generate raw outputs, score
- `--mode score`: Re-score existing raw JSONL files

### 3.2 Prompt Structure

The evaluation script builds prompts automatically:

```
You are given an engineering plot image. Read the plot and answer the question.

Question:
[Question from dataset]

Return ONLY a single JSON object matching this schema (numbers or null; no strings; no units; no extra keys):
{
  "field1": <number or null>  // units,
  "field2": <number or null>  // units,
  ...
}

Notes:
- Use cp_* fields as intermediate plot reads (checkpoints) that help verify your understanding of the plot.
- If you cannot determine a value, output null for that key.
- IMPORTANT: do NOT output arithmetic expressions like 1025/615; output a decimal number.
```

**Key Points**:
- Schema includes ALL fields from `final_fields + checkpoint_fields`
- Units hints provided in comments
- Models can output `null` if value cannot be determined
- No specific computation guidance (models must infer from field names)

### 3.3 JSON Extraction

The evaluator uses robust JSON extraction:
1. Tries direct JSON parse
2. Extracts from fenced code blocks (```json ... ```)
3. Finds first `{...}` blob
4. Sanitizes fractions (e.g., "1025/615" → decimal)
5. Removes trailing commas

### 3.4 Scoring Logic

For each (item, field) pair:

1. **Extract prediction**: From model's parsed JSON
2. **Get ground truth**: From dataset `ground_truth`
3. **Compute errors**:
   - Absolute error: `|pred - gold|`
   - Relative error: `|pred - gold| / max(|gold|, 1e-12)`
4. **Get tolerances**: From tolerance map (family-specific or heuristic)
5. **Pass/fail**: `pass = (abs_err <= abs_tol) OR (rel_err <= rel_tol)`

### 3.5 Tolerance Policies

**Two policies available**:

1. **`plotread`** (default): Human-friendly tolerances
   - Simulates what a human could reasonably read from plots
   - More forgiving for visual estimation errors
   - Example: `step_response.percent_overshoot`: ±2.5 absolute OR ±7% relative

2. **`strict`**: Tighter tolerances
   - More precise requirements
   - Example: `step_response.percent_overshoot`: ±2.0 absolute OR ±5% relative

**Tolerance Sources**:
- Explicit definitions for all known families
- Heuristic fallback for unseen fields (based on field name patterns)
- Field suffixes (`_hz`, `_db`, `_deg`, `_s`, `_v`, etc.) inform heuristics

---

## 4. Output Files Explained

### 4.1 Raw Output File

**File**: `raw_<provider>_<model>.jsonl`

**Format**: One JSON record per item

```json
{
  "provider": "openai",
  "model": "gpt-4.1",
  "id": "step_response_000",
  "type": "step_response",
  "image_path": "/path/to/image.png",
  "prompt": "[full prompt text]",
  "raw_text": "[model's raw text output]",
  "parsed_json": {
    "percent_overshoot": 52.0,
    "settling_time_s": 5.0,
    "steady_state": 1.0,
    "cp_peak_time_s": 0.85,
    ...
  },
  "latency_s": 3.86,
  "error": null
}
```

**Purpose**: Preserves raw outputs for audit, debugging, and reproducibility.

### 4.2 Per-Item CSV

**File**: `per_item.csv`

**Structure**: One row per (item, field) combination

**Columns**:
- `provider`: API provider (e.g., "openai")
- `model`: Model name (e.g., "gpt-4.1")
- `id`: Item ID (e.g., "step_response_000")
- `type`: Plot family
- `field`: Field name (e.g., "percent_overshoot")
- `is_checkpoint`: Boolean (True for cp_* fields)
- `pred`: Model's prediction (or null)
- `gold`: Ground truth value
- `abs_err`: Absolute error `|pred - gold|`
- `rel_err`: Relative error `|pred - gold| / |gold|`
- `abs_tol`: Absolute tolerance threshold
- `rel_tol`: Relative tolerance threshold
- `pass`: Boolean (True if within tolerance)
- `latency_s`: API call latency (seconds)
- `error`: Error message if API call failed (or null)

**Use Case**: Detailed analysis of individual predictions.

### 4.3 Item-Level CSV

**File**: `item_level.csv`

**Structure**: One row per item (aggregated across fields)

**Columns**:
- `provider`, `model`, `id`, `type`
- `n_fields`: Total number of fields for this item
- `final_n`: Number of final (non-checkpoint) fields
- `checkpoint_n`: Number of checkpoint fields
- `all_final_pass`: True if ALL final fields passed
- `all_checkpoint_pass`: True if ALL checkpoint fields passed

**Use Case**: Overall item-level performance (did model get everything right?).

### 4.4 Summary CSV

**File**: `summary.csv`

**Structure**: One row per (model, type, scope, field) combination

**Columns**:
- `provider`, `model`, `type`, `scope` ("final" or "checkpoint"), `field`
- `n`: Number of items with this field
- `pass_rate`: Fraction that passed (0.0 to 1.0)
- `mean_abs_err`: Mean absolute error
- `median_abs_err`: Median absolute error
- `mean_rel_err`: Mean relative error
- `p95_abs_err`: 95th percentile absolute error
- `mean_latency_s`: Mean API latency

**Use Case**: Performance breakdown by family, field, and scope.

### 4.5 Overall CSV

**File**: `overall.csv`

**Structure**: One row per model (aggregated across all items)

**Columns**:
- `provider`, `model`
- `n`: Total number of (item, field) pairs evaluated
- `overall_pass_rate`: Overall pass rate (all fields)
- `mean_abs_err`: Mean absolute error (all fields)
- `mean_rel_err`: Mean relative error (all fields)
- `mean_latency_s`: Mean API latency
- `final_n`: Number of final field evaluations
- `final_pass_rate`: Pass rate for final fields only
- `checkpoint_n`: Number of checkpoint field evaluations
- `checkpoint_pass_rate`: Pass rate for checkpoint fields only

**Use Case**: High-level model comparison.

---

## 5. Key Metrics and Interpretation

### 5.1 Pass Rate

**Definition**: Fraction of (item, field) pairs where `abs_err <= abs_tol OR rel_err <= rel_tol`

**Interpretation**:
- **High pass rate (>80%)**: Model can reliably read plots
- **Medium pass rate (50-80%)**: Model has partial understanding
- **Low pass rate (<50%)**: Model struggles with plot reading

**By Scope**:
- **Final fields**: Primary answers (most important)
- **Checkpoint fields**: Intermediate reads (verification)

### 5.2 Error Metrics

**Absolute Error**: `|pred - gold|`
- Units: Same as the field (Hz, dB, MPa, etc.)
- Interpretation: Direct measure of accuracy

**Relative Error**: `|pred - gold| / |gold|`
- Units: Dimensionless (fraction or percentage)
- Interpretation: Normalized accuracy (useful for comparing fields with different scales)

**95th Percentile Error**: Captures worst-case performance
- Useful for identifying problematic fields

### 5.3 Latency

**Mean Latency**: Average API call time
- Important for practical deployment
- Varies by provider and model

### 5.4 Common Failure Modes

1. **Null outputs**: Model outputs `null` (cannot determine value)
   - Check: `pred IS NULL` in per_item.csv
   - May indicate: Plot too difficult, unclear question, or model limitation

2. **Large absolute errors**: Model reads wrong value
   - Check: `abs_err` column
   - May indicate: Misreading axes, wrong units, or confusion

3. **Systematic bias**: Consistent over/under-estimation
   - Check: `pred - gold` distribution
   - May indicate: Systematic misunderstanding

4. **Checkpoint failures**: Final fields pass but checkpoints fail
   - Check: `all_final_pass` vs `all_checkpoint_pass`
   - May indicate: Model gets right answer but wrong reasoning

---

## 6. Analysis Workflow

### 6.1 Initial Analysis

1. **Check Overall Performance** (`overall.csv`)
   - Compare models by `overall_pass_rate`
   - Check `final_pass_rate` vs `checkpoint_pass_rate`
   - Review `mean_latency_s` for practical considerations

2. **Identify Problem Families** (`summary.csv`)
   - Filter by `type` and `scope="final"`
   - Sort by `pass_rate` ascending
   - Identify families with low pass rates

3. **Examine Problem Fields** (`summary.csv`)
   - Filter by problematic family
   - Check `pass_rate` by `field`
   - Review `mean_abs_err` and `p95_abs_err`

### 6.2 Deep Dive Analysis

1. **Per-Item Analysis** (`per_item.csv`)
   - Filter by problematic family/field
   - Check `pred` vs `gold` values
   - Identify patterns (systematic errors, null outputs)

2. **Difficulty Analysis**
   - Cross-reference with `generation.difficulty` in JSONL
   - Compare performance: clean vs moderate vs edge
   - Expected: Performance degrades with difficulty

3. **Error Distribution**
   - Histogram of `abs_err` by family
   - Identify outliers (very large errors)
   - Check if errors are within tolerance bounds

### 6.3 Checkpoint Analysis

1. **Checkpoint vs Final** (`item_level.csv`)
   - Compare `all_final_pass` vs `all_checkpoint_pass`
   - If checkpoints fail but finals pass: Model may be guessing
   - If both fail: Model cannot read plot

2. **Checkpoint Field Performance** (`summary.csv`)
   - Filter by `scope="checkpoint"`
   - Identify which checkpoint fields are hardest
   - May reveal specific reading challenges

---

## 7. Example Analysis Queries

### 7.1 Overall Model Comparison

```python
import pandas as pd

# Load overall results
overall = pd.read_csv('results/gpt41_plotread/overall.csv')
print(overall[['model', 'overall_pass_rate', 'final_pass_rate', 'mean_latency_s']])
```

### 7.2 Family Performance Breakdown

```python
# Load summary
summary = pd.read_csv('results/gpt41_plotread/summary.csv')

# Final fields only, by family
final_summary = summary[summary['scope'] == 'final']
family_perf = final_summary.groupby('type')['pass_rate'].mean().sort_values()
print(family_perf)
```

### 7.3 Error Analysis

```python
# Load per-item
per_item = pd.read_csv('results/gpt41_plotread/per_item.csv')

# Find items with large errors
large_errors = per_item[per_item['abs_err'] > 10].sort_values('abs_err', ascending=False)
print(large_errors[['id', 'type', 'field', 'pred', 'gold', 'abs_err']])
```

### 7.4 Null Output Analysis

```python
# Count null outputs by family
nulls = per_item[per_item['pred'].isna()]
null_by_family = nulls.groupby('type').size()
print(null_by_family.sort_values(ascending=False))
```

### 7.5 Difficulty Impact

```python
# Load JSONL and merge with results
import json
items = []
with open('data/plotchain_v4/plotchain_v4.jsonl') as f:
    for line in f:
        items.append(json.loads(line))

item_df = pd.DataFrame(items)
item_df['difficulty'] = item_df['generation'].apply(lambda x: x.get('difficulty'))

# Merge with per_item
merged = per_item.merge(item_df[['id', 'difficulty']], on='id')
difficulty_perf = merged.groupby('difficulty')['pass'].mean()
print(difficulty_perf)
```

---

## 8. Expected Results Patterns

### 8.1 Good Performance Indicators

- **Pass rate > 80%** for final fields
- **Low null rate** (< 5%)
- **Errors within tolerance** (not just close)
- **Consistent across families** (no major outliers)
- **Checkpoint fields pass** (verifies understanding)

### 8.2 Warning Signs

- **High null rate** (> 20%): Model cannot read many plots
- **Systematic errors**: Consistent bias in one direction
- **Family-specific failures**: One family much worse than others
- **Checkpoint failures**: Finals pass but checkpoints fail (guessing?)
- **Large p95 errors**: Some predictions very wrong

### 8.3 Difficulty Gradient

Expected performance should decrease with difficulty:
- **Clean**: Highest pass rate (grid, markers, no noise)
- **Moderate**: Medium pass rate (some noise, grid)
- **Edge**: Lowest pass rate (no grid, noise, coarse ticks)

If this pattern doesn't hold, investigate why.

---

## 9. Benchmark Properties

### 9.1 Reproducibility

- **100% deterministic**: Same (seed, family, index) → same plot
- **No randomness in GT**: Computed from parameters
- **Validation**: Baseline recomputation matches stored GT

### 9.2 Fairness

- **Uniform prompts**: All families get same prompt structure
- **No hints**: Models must infer checkpoint field meanings
- **Consistent tolerances**: Same policy across families
- **No bias**: Synthetic plots avoid real-world biases

### 9.3 Coverage

- **15 diverse families**: Control, signal processing, materials, mechanical
- **450 total items**: 30 per family
- **Multiple difficulty levels**: Tests robustness
- **Checkpoint fields**: Verifies intermediate understanding

---

## 10. Common Analysis Tasks

### Task 1: Compare Two Models

```python
# Load both overall CSVs
model1 = pd.read_csv('results/model1/overall.csv')
model2 = pd.read_csv('results/model2/overall.csv')

# Compare pass rates
comparison = pd.merge(
    model1[['model', 'overall_pass_rate', 'final_pass_rate']],
    model2[['model', 'overall_pass_rate', 'final_pass_rate']],
    on='model', suffixes=('_model1', '_model2')
)
print(comparison)
```

### Task 2: Find Hardest Fields

```python
summary = pd.read_csv('results/gpt41_plotread/summary.csv')
final_summary = summary[summary['scope'] == 'final']
hardest = final_summary.nsmallest(10, 'pass_rate')[['type', 'field', 'pass_rate', 'mean_abs_err']]
print(hardest)
```

### Task 3: Analyze Error Patterns

```python
per_item = pd.read_csv('results/gpt41_plotread/per_item.csv')

# Group by field, compute error statistics
error_stats = per_item.groupby(['type', 'field']).agg({
    'abs_err': ['mean', 'median', 'std', 'max'],
    'rel_err': ['mean', 'median'],
    'pass': 'mean'
}).round(3)
print(error_stats)
```

### Task 4: Difficulty Impact Analysis

```python
# Merge difficulty from JSONL
import json
items = {it['id']: it for it in [json.loads(l) for l in open('data/plotchain_v4/plotchain_v4.jsonl')]}

per_item = pd.read_csv('results/gpt41_plotread/per_item.csv')
per_item['difficulty'] = per_item['id'].map(lambda x: items[x]['generation']['difficulty'])

# Performance by difficulty
diff_perf = per_item.groupby('difficulty').agg({
    'pass': 'mean',
    'abs_err': 'mean',
    'pred': lambda x: x.isna().mean()  # null rate
})
print(diff_perf)
```

---

## 11. Interpretation Guidelines

### 11.1 What Constitutes "Good" Performance?

- **Overall pass rate > 80%**: Model can reliably read plots
- **Final fields > 85%**: Primary answers are accurate
- **Checkpoint fields > 75%**: Model understands intermediate values
- **Low null rate (< 5%)**: Model attempts most fields
- **Consistent across families**: No major outliers

### 11.2 What Indicates Problems?

- **Pass rate < 60%**: Model struggles significantly
- **High null rate (> 20%)**: Model gives up frequently
- **Large p95 errors**: Some predictions very wrong
- **Family outliers**: One family much worse (may indicate domain gap)
- **Checkpoint failures**: Model guesses final answers

### 11.3 Comparing Models

When comparing models:
1. **Use same tolerance policy** (plotread or strict)
2. **Compare final fields** (most important)
3. **Check checkpoint fields** (verifies understanding)
4. **Consider latency** (practical deployment)
5. **Examine error distributions** (not just pass rates)

---

## 12. File Locations Reference

### Dataset Files
- `data/plotchain_v4/plotchain_v4.jsonl` - Main dataset
- `data/plotchain_v4/images/<family>/*.png` - Plot images
- `data/plotchain_v4/validation_rows.csv` - Baseline validation
- `data/plotchain_v4/validation_summary.csv` - Validation summary

### Evaluation Scripts
- `run_plotchain_v4_eval.py` - Main evaluation script
- `generate_plotchain_v4.py` - Dataset generation script

### Results Files (in `results/<model>/`)
- `raw_<provider>_<model>.jsonl` - Raw model outputs
- `per_item.csv` - Detailed per-item results
- `item_level.csv` - Item-level aggregates
- `summary.csv` - Summary by type/field
- `overall.csv` - Overall model performance

---

## 13. Key Questions for Analysis

When analyzing results, consider:

1. **Overall Performance**: What's the overall pass rate?
2. **Family Variation**: Which families are easiest/hardest?
3. **Field Difficulty**: Which fields are most challenging?
4. **Difficulty Impact**: Does performance degrade with difficulty?
5. **Error Patterns**: Are errors systematic or random?
6. **Null Rate**: How often does the model give up?
7. **Checkpoint Performance**: Can the model read intermediate values?
8. **Latency**: Is the model fast enough for practical use?
9. **Failure Modes**: What causes failures? (null, large errors, systematic bias)
10. **Comparison**: How does this model compare to others?

---

## 14. Technical Details

### 14.1 Tolerance System

Tolerances are defined per (family, field) pair:
- **Absolute tolerance**: Maximum allowed absolute error
- **Relative tolerance**: Maximum allowed relative error
- **Pass condition**: `(abs_err <= abs_tol) OR (rel_err <= rel_tol)`

Example: `step_response.percent_overshoot` with plotread policy:
- `abs_tol = 2.5`
- `rel_tol = 0.07` (7%)
- Pass if: `|pred - gold| <= 2.5` OR `|pred - gold| / |gold| <= 0.07`

### 14.2 Field Quantization

Ground truth values are quantized to human-friendly precision:
- `percent_overshoot`: 1 decimal place
- `settling_time_s`: 2 decimal places
- `frequency_hz`: 0 decimal places (integers)
- `cp_peak_ratio`: 1 decimal place

This ensures expected outputs are readable (e.g., "52.4" not "52.3874629").

### 14.3 Checkpoint Fields

Checkpoint fields (`cp_*`) serve multiple purposes:
1. **Verification**: Verify model can read intermediate values
2. **Debugging**: Identify where model makes mistakes
3. **Fairness**: Ensure model isn't just guessing final answers

Models must infer checkpoint meanings from field names:
- `cp_peak_ratio`: Ratio of peak magnitudes
- `cp_peak_time_s`: Time at peak
- `cp_band_lower`: Lower bound of tolerance band

No explicit computation guidance is provided (tests understanding).

---

## 15. Usage Instructions

### 15.1 Running Evaluation

```bash
# Basic evaluation
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4.1 \
  --out_dir results/gpt41_plotread \
  --policy plotread

# Multiple models
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4.1 gemini:gemini-2.0-flash-exp \
  --out_dir results/multi_model \
  --policy plotread

# Test run (first 10 items)
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4.1 \
  --out_dir results/test \
  --policy plotread \
  --limit 10
```

### 15.2 Re-scoring Existing Results

```bash
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --mode score \
  --raw_glob "raw_*.jsonl" \
  --out_dir results/gpt41_plotread \
  --policy plotread
```

### 15.3 Regenerating Dataset

```bash
python3 generate_plotchain_v4.py \
  --out_dir data/plotchain_v4 \
  --n_per_family 30 \
  --seed 0
```

---

## 16. Analysis Checklist

When analyzing results, verify:

- [ ] Overall pass rate is reasonable (>70% for good models)
- [ ] Final fields perform better than checkpoints (expected)
- [ ] Performance degrades with difficulty (clean > moderate > edge)
- [ ] No single family dominates failures
- [ ] Null rate is low (<10%)
- [ ] Errors are within tolerance bounds (not just close)
- [ ] Latency is acceptable for deployment
- [ ] Error patterns make sense (not systematic bias)
- [ ] Checkpoint fields show understanding (not guessing)

---

## 17. Contact and References

**Dataset**: PlotChain v4  
**Version**: v4 (FROZEN)  
**Total Items**: 450 (30 per family × 15 families)  
**Ground Truth**: Deterministic (computed from parameters)  
**Validation**: 100% baseline validation pass rate

**Key Files**:
- Generation: `generate_plotchain_v4.py`
- Evaluation: `run_plotchain_v4_eval.py`
- Dataset: `data/plotchain_v4/plotchain_v4.jsonl`

---

---

## 18. Quick Reference for Analysis

### CSV File Meanings

| File | Rows | Key Columns | Use Case |
|------|------|-------------|----------|
| `overall.csv` | 1 per model | `overall_pass_rate`, `final_pass_rate`, `mean_latency_s` | High-level model comparison |
| `summary.csv` | 1 per (model, type, scope, field) | `pass_rate`, `mean_abs_err`, `p95_abs_err` | Performance by family/field |
| `item_level.csv` | 1 per item | `all_final_pass`, `all_checkpoint_pass` | Item-level success rates |
| `per_item.csv` | 1 per (item, field) | `pred`, `gold`, `abs_err`, `rel_err`, `pass` | Detailed error analysis |

### Key Metrics to Check

1. **Overall Pass Rate** (`overall.csv.overall_pass_rate`)
   - Target: > 80% for good models
   - Interpretation: Fraction of all predictions that are correct

2. **Final vs Checkpoint** (`overall.csv.final_pass_rate` vs `checkpoint_pass_rate`)
   - Expected: Final > Checkpoint (checkpoints are harder)
   - Warning: If checkpoint much lower, model may be guessing

3. **Null Rate** (`per_item.csv` where `pred IS NULL`)
   - Target: < 5% for good models
   - Interpretation: How often model gives up

4. **95th Percentile Error** (`summary.csv.p95_abs_err`)
   - Interpretation: Worst-case performance
   - Use: Identify problematic fields

### Common Analysis Patterns

**Pattern 1: Model Comparison**
```python
overall = pd.read_csv('results/model1/overall.csv')
print(overall[['model', 'overall_pass_rate', 'final_pass_rate']])
```

**Pattern 2: Family Breakdown**
```python
summary = pd.read_csv('results/model1/summary.csv')
family_perf = summary[summary['scope']=='final'].groupby('type')['pass_rate'].mean()
print(family_perf.sort_values())
```

**Pattern 3: Error Analysis**
```python
per_item = pd.read_csv('results/model1/per_item.csv')
large_errors = per_item.nlargest(20, 'abs_err')[['id', 'type', 'field', 'pred', 'gold', 'abs_err']]
print(large_errors)
```

**Pattern 4: Null Analysis**
```python
per_item = pd.read_csv('results/model1/per_item.csv')
null_rate = per_item['pred'].isna().mean()
null_by_family = per_item[per_item['pred'].isna()].groupby('type').size()
print(f"Overall null rate: {null_rate:.1%}")
print(null_by_family)
```

---

**End of Context Document**

This document provides complete context for analyzing PlotChain v4 evaluation results. Use it to understand the benchmark structure, evaluation methodology, and how to interpret the CSV outputs.

**For Google Gemini**: Use this document to understand the PlotChain v4 benchmark when analyzing evaluation results. The document explains what each CSV file contains, what metrics mean, and how to interpret model performance.

