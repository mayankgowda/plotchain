# Re-Run Evaluations with Temperature=0

## Status

**Previous runs**: Done WITHOUT temperature=0 enforced  
**Current code**: Temperature=0 now enforced  
**Action Required**: ✅ **YES - Re-run all evaluations**

## Why Re-Run?

1. **Reproducibility**: Temperature=0 ensures deterministic results
2. **Methodology Checklist**: Required for "irrefutable" methodology
3. **Paper Standards**: Deterministic results are expected for publication
4. **Consistency**: All models should use same temperature setting

## Commands to Re-Run

### GPT-4.1
```bash
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4.1 \
  --out_dir results/gpt41_plotread_temp0 \
  --mode run \
  --policy plotread \
  --max_output_tokens 2000 \
  --concurrent 10 \
  --overwrite
```

### Claude Sonnet 4.5
```bash
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models anthropic:claude-sonnet-4-5-20250929 \
  --out_dir results/claudesonnet45_plotread_temp0 \
  --mode run \
  --policy plotread \
  --max_output_tokens 2000 \
  --concurrent 10 \
  --overwrite
```

### Gemini 2.5 Pro
```bash
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.5-pro \
  --out_dir results/gemini25pro_plotread_temp0 \
  --mode run \
  --policy plotread \
  --max_output_tokens 2000 \
  --concurrent 10 \
  --overwrite
```

### GPT-4o
```bash
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4o \
  --out_dir results/gpt4o_plotread_temp0 \
  --mode run \
  --policy plotread \
  --max_output_tokens 2000 \
  --concurrent 10 \
  --overwrite
```

## Cost Estimate

- **Items**: 450 per model
- **Models**: 4
- **Total API Calls**: 1,800
- **Estimated Cost**: $50-100 (depending on providers)
- **Time**: 20-45 minutes with concurrent requests

## Verification

After re-running, verify temperature=0 was used:

```bash
# Check that temperature=0 is in the code
grep -n "temperature=0" run_plotchain_v4_eval.py

# Verify results are deterministic (re-run a few items and compare)
```

## Alternative Approach

If cost is a concern, you could:

1. **Re-run subset**: Re-run 1-2 models to verify temperature=0 works
2. **Document**: Note in paper that all models will be re-run with temperature=0 before publication
3. **Compare**: Check if results differ significantly (they shouldn't if APIs defaulted to low temperature)

However, for "irrefutable" methodology, **all models should be re-run**.

## Expected Impact

- **Results**: Should be very similar (APIs may have defaulted to low temperature)
- **Determinism**: Results will be fully reproducible
- **Methodology**: Meets "irrefutable" checklist requirements

---

**Recommendation**: ✅ **Re-run all evaluations with temperature=0**

