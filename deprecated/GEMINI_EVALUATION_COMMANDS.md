# Gemini Model Evaluation Commands

## Gemini 2.0 Pro / Gemini 2.5 Pro Evaluation

**Note**: Google's Gemini model naming may vary. Try these commands in order:

---

## Option 1: Gemini 2.0 Flash (Recommended - Fastest)

```bash
# Test with 1 item first
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.0-flash-exp \
  --out_dir results/gemini20_test \
  --policy plotread \
  --limit 1

# If successful, run full evaluation
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.0-flash-exp \
  --out_dir results/gemini20_plotread \
  --policy plotread
```

---

## Option 2: Gemini 1.5 Pro (Most Likely "Pro" Model)

```bash
# Test with 1 item first
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-1.5-pro \
  --out_dir results/gemini15pro_test \
  --policy plotread \
  --limit 1

# If successful, run full evaluation
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-1.5-pro \
  --out_dir results/gemini15pro_plotread \
  --policy plotread
```

---

## Option 3: Gemini 1.5 Pro Latest

```bash
# Test with 1 item first
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-1.5-pro-latest \
  --out_dir results/gemini15pro_test \
  --policy plotread \
  --limit 1

# If successful, run full evaluation
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-1.5-pro-latest \
  --out_dir results/gemini15pro_plotread \
  --policy plotread
```

---

## Option 4: Gemini 2.0 Flash Thinking (If Available)

```bash
# Test with 1 item first
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.0-flash-thinking-exp \
  --out_dir results/gemini20_test \
  --policy plotread \
  --limit 1

# If successful, run full evaluation
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.0-flash-thinking-exp \
  --out_dir results/gemini20_plotread \
  --policy plotread
```

---

## How to Find the Correct Model Name

1. **Check Google AI Studio**: https://aistudio.google.com/app/apikey
2. **Check Gemini API Docs**: https://ai.google.dev/gemini-api/docs/models
3. **List available models** (if API supports it):
   ```python
   import google.generativeai as genai
   genai.configure(api_key="YOUR_API_KEY")
   for model in genai.list_models():
       if 'gemini' in model.name.lower():
           print(model.name)
   ```

---

## Recommended Approach

1. **Start with**: `gemini:gemini-2.0-flash-exp` (fastest, latest)
2. **If that fails**: Try `gemini:gemini-1.5-pro` (most stable Pro model)
3. **If that fails**: Check Google's API documentation for current model names

---

## Full Evaluation Command (Once Model Name Confirmed)

```bash
# Full evaluation with rate limiting (recommended)
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.0-flash-exp \
  --out_dir results/gemini20_plotread \
  --policy plotread \
  --sleep_s 0.5
```

**Expected Time**: 2-3 hours (450 items × ~3-5s latency)  
**Expected Cost**: ~$10-20 (Gemini is typically cheaper than GPT-4)

---

## Troubleshooting

**If you get "model not found" error**:
1. Check your `GOOGLE_API_KEY` environment variable is set
2. Verify the model name in Google's documentation
3. Try alternative model names listed above

**If you get rate limit errors**:
- Add `--sleep_s 1.0` to slow down requests
- Or use `--limit 10` to test with fewer items first

---

## Verification

After running, check:
```bash
# Check overall results
cat results/gemini20_plotread/overall.csv

# Should show non-zero pass rate (expect 75-85%)
```

