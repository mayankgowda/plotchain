# Concurrent Evaluation Usage Guide

## Overview

Concurrent evaluation with rate limiting has been implemented for all providers (OpenAI, Anthropic, Gemini). This significantly speeds up evaluation, especially for slow models like Gemini 2.5 Pro (30s per request → ~20-45 minutes with concurrency).

---

## Features

✅ **Automatic concurrency**: Enabled by default for all providers  
✅ **Rate limiting**: Provider-specific limits to avoid API errors  
✅ **Thread-safe**: Safe concurrent file writing  
✅ **Progress tracking**: Real-time progress bar  
✅ **Fallback**: Automatically falls back to sequential if concurrent fails  
✅ **Backward compatible**: Use `--sequential` to disable concurrency

---

## Rate Limits (Configurable in `concurrent_eval.py`)

| Provider | Requests/Min | Max Concurrent | Notes |
|----------|--------------|----------------|-------|
| **OpenAI** | 60 | 10 | Conservative for paid tier |
| **Anthropic** | 50 | 10 | Conservative for paid tier |
| **Gemini** | 60 | 10 | Conservative for paid tier |

**Location**: `concurrent_eval.py` → `PROVIDER_RATE_LIMITS` dictionary

---

## Usage

### Default (Concurrent - Recommended)

```bash
# Concurrent evaluation (automatic)
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.5-pro \
  --out_dir results/gemini25pro_plotread \
  --policy plotread
```

**Behavior**: Uses provider-specific concurrent workers (default: 10 for all providers)

### Custom Concurrency Level

```bash
# Use 5 concurrent workers instead of default
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.5-pro \
  --out_dir results/gemini25pro_plotread \
  --policy plotread \
  --concurrent 5
```

**Note**: Will be capped at provider's `max_concurrent` limit

### Sequential (Disable Concurrency)

```bash
# Use sequential processing (old behavior)
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.5-pro \
  --out_dir results/gemini25pro_plotread \
  --policy plotread \
  --sequential
```

---

## Performance Comparison

### Gemini 2.5 Pro Example

| Mode | Workers | Time (450 items) | Speedup |
|------|---------|------------------|---------|
| Sequential | 1 | **3.75 hours** | 1x |
| Concurrent (5) | 5 | **45 minutes** | 5x |
| Concurrent (10) | 10 | **22.5 minutes** | 10x |

**Recommendation**: Use default (10 workers) unless you hit rate limits

---

## How It Works

1. **Rate Limiter**: Token bucket algorithm limits requests per minute
2. **Thread Pool**: `ThreadPoolExecutor` manages concurrent workers
3. **Progress Tracking**: `tqdm` shows real-time progress
4. **Thread-Safe Writing**: Results written sequentially to avoid corruption
5. **Error Handling**: Failed requests are logged, evaluation continues

---

## Adjusting Rate Limits

Edit `concurrent_eval.py`:

```python
PROVIDER_RATE_LIMITS: Dict[str, RateLimit] = {
    "openai": RateLimit(
        requests_per_minute=60,  # Increase if you have higher tier
        max_concurrent=10,        # Increase for faster evaluation
    ),
    "anthropic": RateLimit(
        requests_per_minute=50,
        max_concurrent=10,
    ),
    "gemini": RateLimit(
        requests_per_minute=60,
        max_concurrent=10,
    ),
}
```

**Tips**:
- **Higher tier accounts**: Increase `requests_per_minute` (check API docs)
- **Faster evaluation**: Increase `max_concurrent` (but watch for rate limits)
- **Rate limit errors**: Decrease both values

---

## Troubleshooting

### Rate Limit Errors (429)

**Symptoms**: Frequent 429 errors, requests failing

**Solutions**:
1. **Reduce concurrency**: `--concurrent 5` or `--concurrent 3`
2. **Reduce rate limit**: Edit `concurrent_eval.py`, lower `requests_per_minute`
3. **Use sequential**: `--sequential` flag

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'concurrent_eval'`

**Solution**: Ensure `concurrent_eval.py` is in the same directory as `run_plotchain_v4_eval.py`

### Fallback to Sequential

**Message**: `[warning] Concurrent evaluation failed: ..., falling back to sequential`

**Cause**: Concurrent evaluation encountered an error

**Action**: Check error message, fix if needed, or use `--sequential` flag

---

## Example Commands

### Gemini 2.5 Pro (Slow Model - Use Concurrency)

```bash
# Fast evaluation with concurrency
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.5-pro \
  --out_dir results/gemini25pro_plotread \
  --policy plotread

# Expected: ~20-45 minutes (vs 3.75 hours sequential)
```

### GPT-4.1 (Fast Model - Concurrency Still Helps)

```bash
# Concurrent evaluation
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4.1 \
  --out_dir results/gpt41_plotread \
  --policy plotread

# Expected: ~10-15 minutes (vs ~1 hour sequential)
```

### Multiple Models

```bash
# Evaluate multiple models concurrently
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4.1 anthropic:claude-sonnet-4-5-20250929 gemini:gemini-2.5-pro \
  --out_dir results/multi_model \
  --policy plotread

# Each model uses its own concurrent workers
```

---

## Technical Details

### Rate Limiter Algorithm

**Token Bucket**:
- Tokens refill at `requests_per_minute / 60` per second
- Each request consumes 1 token
- Blocks if no tokens available (waits until token available)

### Thread Safety

- **Rate limiter**: Thread-safe (uses `threading.Lock`)
- **File writing**: Sequential (writes in completion order)
- **Progress bar**: Thread-safe (`tqdm` handles concurrency)

### Error Handling

- **API errors**: Caught and logged, evaluation continues
- **Rate limit errors**: Handled by rate limiter (waits)
- **Import errors**: Falls back to sequential processing

---

## Summary

✅ **Concurrent evaluation is enabled by default**  
✅ **Significantly faster** (5-10x speedup for slow models)  
✅ **Rate limited** to avoid API errors  
✅ **Backward compatible** (use `--sequential` to disable)  
✅ **Easy to configure** (edit `concurrent_eval.py`)

**For Gemini 2.5 Pro**: Use default concurrent mode (10 workers) for ~20-45 minute evaluation instead of 3.75 hours!

