# Concurrent Requests Recommendation for Gemini 2.5 Pro

**Current Situation**: 30s per request × 450 items = **3.75 hours** (very slow)  
**Question**: Should we use concurrent requests?

---

## Recommendation: ✅ **YES, but with caution**

### **Recommended Approach: Moderate Concurrency (5-10 concurrent requests)**

**Rationale**:
- ✅ **Significant time savings**: 3.75 hours → ~20-40 minutes (5-10x speedup)
- ✅ **Paid account**: Should have higher rate limits
- ✅ **Manageable risk**: 5-10 concurrent requests is reasonable
- ⚠️ **Rate limit handling**: Need to implement backoff/retry logic

---

## Time Savings Analysis

### Current (Sequential)
- **Time**: 450 items × 30s = **13,500 seconds = 3.75 hours**
- **Rate**: 1 request per 30s = 2 requests/minute

### With Concurrency

| Concurrent Requests | Time per Batch | Total Time | Speedup |
|---------------------|----------------|------------|---------|
| **1 (current)** | 30s | **3.75 hours** | 1x |
| **5 concurrent** | 30s | **45 minutes** | 5x |
| **10 concurrent** | 30s | **22.5 minutes** | 10x |
| **20 concurrent** | 30s | **11.25 minutes** | 20x |

**Recommendation**: **5-10 concurrent requests** (balance between speed and reliability)

---

## Gemini API Rate Limits (Estimated)

### Free Tier
- **Requests per minute**: ~15 RPM
- **Concurrent requests**: Limited
- **Not suitable**: For concurrent requests

### Paid Tier (Your Case)
- **Requests per minute**: ~60-100 RPM (estimated)
- **Concurrent requests**: Usually allowed (check docs)
- **Suitable**: For moderate concurrency (5-10)

**Action**: Check Gemini API documentation for exact paid tier limits:
- https://ai.google.dev/gemini-api/docs/quota
- Look for "requests per minute" and "concurrent requests"

---

## Implementation Considerations

### 1. **Concurrency Level**

**Recommended**: **5-10 concurrent requests**

**Why**:
- ✅ **Safe**: Low risk of hitting rate limits
- ✅ **Effective**: 5-10x speedup (45-22 minutes)
- ✅ **Reliable**: Less likely to cause errors
- ⚠️ **Not too aggressive**: Avoids overwhelming API

**Not Recommended**: **20+ concurrent**
- ⚠️ **Risk**: May hit rate limits
- ⚠️ **Errors**: More likely to fail
- ⚠️ **Unnecessary**: Diminishing returns

### 2. **Error Handling**

**Critical**: Implement robust error handling

**Required**:
- ✅ **Rate limit detection**: Check for 429 errors
- ✅ **Exponential backoff**: Retry with increasing delays
- ✅ **Request queuing**: Queue requests if rate limited
- ✅ **Progress tracking**: Save progress periodically
- ✅ **Resume capability**: Can resume from failures

### 3. **Progress Persistence**

**Important**: Save progress frequently

**Why**:
- ⚠️ **Long runtime**: Even with concurrency, 20-45 minutes
- ⚠️ **Failure risk**: Concurrent requests can fail
- ✅ **Resume**: Can restart from last successful item

**Implementation**:
- Save raw JSONL after each batch (every 10-20 items)
- Use `--overwrite` carefully (don't lose progress)

### 4. **Monitoring**

**Recommended**: Add monitoring/logging

**Track**:
- ✅ **Success rate**: % of requests that succeed
- ✅ **Error rate**: % that fail (429, timeouts, etc.)
- ✅ **Average latency**: Track if latency increases
- ✅ **Rate limit hits**: Count 429 errors

---

## Recommended Implementation Strategy

### Phase 1: Test with Small Batch

**Start conservative**:
```python
# Test with 10 items, 3 concurrent requests
--limit 10
--concurrent 3  # (if implemented)
```

**Verify**:
- ✅ No rate limit errors
- ✅ All requests succeed
- ✅ Latency is stable (~30s per request)

### Phase 2: Scale Up Gradually

**If Phase 1 succeeds**:
1. **Try 5 concurrent** with 50 items
2. **If stable**: Try 10 concurrent with 100 items
3. **If stable**: Run full 450 items with 10 concurrent

**If rate limited**:
- Reduce to 3-5 concurrent
- Add longer delays between batches

### Phase 3: Full Run

**Once stable**:
- Run full 450 items with 5-10 concurrent
- Monitor for errors
- Save progress frequently

---

## Code Changes Required

### Current Code (Sequential)
```python
for item in items:
    result = call_model(...)  # Blocks for 30s
    save_result(result)
```

### Proposed Code (Concurrent)
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_item(item):
    try:
        result = call_model(...)
        return (item['id'], result, None)
    except Exception as e:
        return (item['id'], None, str(e))

# Process with concurrency
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_item, item): item for item in items}
    
    for future in as_completed(futures):
        item_id, result, error = future.result()
        if error:
            # Handle error (retry, log, etc.)
            handle_error(item_id, error)
        else:
            save_result(result)
        
        # Rate limit handling
        if rate_limit_detected:
            time.sleep(backoff_delay)
```

---

## Risk Assessment

### Low Risk ✅
- **5 concurrent requests**: Very safe, recommended
- **10 concurrent requests**: Generally safe, recommended
- **Proper error handling**: Essential

### Medium Risk ⚠️
- **15-20 concurrent**: May hit rate limits
- **No error handling**: Can lose progress
- **No progress saving**: Can't resume

### High Risk ❌
- **30+ concurrent**: Very likely to hit limits
- **No backoff**: Will fail repeatedly
- **No resume**: Must restart from beginning

---

## Alternative: Batch Processing

### Option: Process in Batches

**Strategy**: Process 50 items at a time with 5-10 concurrent

**Benefits**:
- ✅ **Progress saving**: Save after each batch
- ✅ **Error recovery**: Can retry failed batches
- ✅ **Monitoring**: Check progress between batches
- ✅ **Rate limit friendly**: Less aggressive

**Implementation**:
```python
batch_size = 50
concurrent = 10

for batch_start in range(0, len(items), batch_size):
    batch = items[batch_start:batch_start + batch_size]
    process_batch_concurrent(batch, concurrent)
    save_progress()  # Save after each batch
    time.sleep(5)  # Brief pause between batches
```

---

## Cost Considerations

### Sequential (Current)
- **Time**: 3.75 hours
- **Cost**: Same API cost
- **Opportunity cost**: High (waiting time)

### Concurrent (5-10)
- **Time**: 20-45 minutes
- **Cost**: Same API cost (same number of requests)
- **Opportunity cost**: Low (much faster)

**Verdict**: ✅ **No additional cost, significant time savings**

---

## Final Recommendation

### ✅ **YES - Use Concurrent Requests**

**Recommended Settings**:
- **Concurrency**: **5-10 requests** (start with 5, scale to 10 if stable)
- **Error handling**: **Essential** (rate limit detection, backoff, retry)
- **Progress saving**: **Frequent** (every 10-20 items)
- **Monitoring**: **Track success/error rates**

**Expected Results**:
- **Time**: 3.75 hours → **20-45 minutes** (5-10x speedup)
- **Cost**: Same (no additional API cost)
- **Risk**: Low (with proper error handling)

### Implementation Priority

1. **High Priority**: Error handling (rate limits, retries)
2. **High Priority**: Progress saving (resume capability)
3. **Medium Priority**: Monitoring/logging
4. **Low Priority**: Advanced features (adaptive concurrency)

---

## Testing Plan

### Step 1: Small Test (10 items, 3 concurrent)
- Verify no rate limits
- Check error handling works
- Confirm progress saving

### Step 2: Medium Test (50 items, 5 concurrent)
- Verify stability
- Check latency consistency
- Confirm no errors

### Step 3: Full Run (450 items, 5-10 concurrent)
- Monitor for rate limits
- Save progress frequently
- Be ready to reduce concurrency if needed

---

## Conclusion

**Recommendation**: ✅ **Use 5-10 concurrent requests**

**Benefits**:
- ✅ **5-10x speedup** (3.75 hours → 20-45 minutes)
- ✅ **No additional cost**
- ✅ **Manageable risk** (with proper error handling)

**Requirements**:
- ✅ **Error handling**: Rate limit detection, backoff, retry
- ✅ **Progress saving**: Frequent saves, resume capability
- ✅ **Monitoring**: Track success/error rates

**Next Steps**:
1. Check Gemini API rate limits for paid tier
2. Implement concurrent processing with error handling
3. Test with small batch first
4. Scale up gradually
5. Run full evaluation

---

**Bottom Line**: With proper error handling, concurrent requests are **highly recommended** for Gemini 2.5 Pro evaluation. The time savings (3.75 hours → 20-45 minutes) justify the implementation effort.

