# Commands to Regenerate Bandpass and Re-run Evaluations

## Step 1: Regenerate Bandpass Plots

```bash
# Regenerate only bandpass_response plots with fixed code
python3 regenerate_bandpass.py \
  --out_dir data/plotchain_v4 \
  --n_per_family 30 \
  --seed 0
```

This will:
- Load existing JSONL (450 items)
- Remove old bandpass_response items (30 items)
- Generate new bandpass_response items with fixed code (30 items)
- Update plotchain_v4.jsonl (450 items total)
- Regenerate bandpass images in `data/plotchain_v4/images/bandpass_response/`

## Step 2: Re-run Evaluations for All Models

### GPT-4.1

```bash
python3 eval_bandpass_only.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --out_dir results/gpt41_plotread \
  --models openai:gpt-4.1 \
  --policy plotread \
  --overwrite \
  --concurrent 10
```

### Claude Sonnet 4.5

```bash
python3 eval_bandpass_only.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --out_dir results/claudesonnet45_plotread \
  --models anthropic:claude-sonnet-4-5-20250929 \
  --policy plotread \
  --overwrite \
  --concurrent 10
```

### Gemini 2.5 Pro

```bash
python3 eval_bandpass_only.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --out_dir results/gemini25pro_plotread \
  --models gemini:gemini-2.5-pro \
  --policy plotread \
  --overwrite \
  --concurrent 30
```

## What These Scripts Do

### regenerate_bandpass.py
1. Loads existing `plotchain_v4.jsonl`
2. Removes all `bandpass_response` items
3. Generates new `bandpass_response` items with **fixed code**
4. Updates `plotchain_v4.jsonl` with corrected items
5. Regenerates bandpass images

### eval_bandpass_only.py
1. Loads existing result files (`per_item.csv`, `raw_*.jsonl`)
2. Removes all `bandpass_response` rows
3. Re-evaluates only `bandpass_response` items
4. Merges new results with existing non-bandpass results
5. Regenerates all report files (`per_item.csv`, `item_level.csv`, `overall.csv`, `summary.csv`)

## Expected Changes

After regeneration and re-evaluation:

1. **Ground truth values will change**:
   - `cp_f1_3db_hz`: Will be closer to actual -3dB points
   - `cp_f2_3db_hz`: Will be closer to actual -3dB points
   - `bandwidth_hz`: Will be significantly different (e.g., 182.7 → 81.7 Hz)
   - `cp_q_factor`: Will change (bandwidth changes)

2. **Model performance may improve**:
   - Previous failures may have been due to wrong ground truth
   - Models may have been reading correctly, but GT didn't match
   - Visual alignment should now be correct (vertical lines match horizontal -3dB line)

3. **Results files updated**:
   - `per_item.csv`: Only bandpass rows updated
   - `overall.csv`: Overall stats recalculated
   - `summary.csv`: Bandpass family stats updated
   - `item_level.csv`: Bandpass item-level stats updated

## Verification

After re-running, check:

```bash
# Check bandpass performance
python3 << 'EOF'
import pandas as pd

df = pd.read_csv('results/gpt41_plotread/per_item.csv')
bandpass = df[df['type'] == 'bandpass_response']
print(f"Bandpass items: {len(bandpass)}")
print(f"Pass rate: {bandpass['pass'].mean():.1%}")
print(f"\nBy field:")
print(bandpass.groupby('field')['pass'].mean().sort_values())
EOF
```

## Notes

- Use `--overwrite` to update existing files
- Use `--concurrent` for faster evaluation (recommended)
- Use `--sequential` to disable concurrency if needed
- The scripts preserve all non-bandpass results

