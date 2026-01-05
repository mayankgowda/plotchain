from __future__ import annotations

"""
Concurrent evaluation with rate limiting for all providers.

This module provides concurrent request processing with provider-specific rate limits.
"""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# -------------------------
# Rate Limit Constants (per provider)
# -------------------------

@dataclass(frozen=True)
class RateLimit:
    """Rate limit configuration for a provider."""
    requests_per_minute: int
    max_concurrent: int  # Maximum concurrent requests


# Rate limits for each provider (conservative defaults for paid accounts)
PROVIDER_RATE_LIMITS: Dict[str, RateLimit] = {
    "openai": RateLimit(
        requests_per_minute=60,  # Conservative: OpenAI paid tier allows more
        max_concurrent=10,
    ),
    "anthropic": RateLimit(
        requests_per_minute=20,  # Conservative: Anthropic paid tier allows more
        max_concurrent=2,
    ),
    "gemini": RateLimit(
        requests_per_minute=40,  # Conservative: Gemini paid tier allows more
        max_concurrent=20,
    ),
}


# -------------------------
# Simple Rate Limiter (Token Bucket)
# -------------------------

class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_refill = time.time()
        self.lock = threading.Lock()
        self.min_interval = 60.0 / requests_per_minute  # Minimum seconds between requests
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        self.tokens = min(self.requests_per_minute, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def acquire(self):
        """Acquire a token, blocking if necessary."""
        with self.lock:
            self._refill()
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # Need to wait
            wait_time = (1.0 - self.tokens) * self.min_interval
            if wait_time > 0:
                time.sleep(wait_time)
            self.tokens = 0.0


# -------------------------
# Concurrent Evaluation
# -------------------------

def process_item(
    item: Dict[str, Any],
    spec: Any,  # ModelSpec
    images_root: Optional[Path],
    jsonl_path: Path,
    max_output_tokens: int,
    call_model_fn: Callable,
    build_prompt_fn: Callable,
    resolve_image_path_fn: Callable,
    get_item_id_fn: Callable,
    get_item_type_fn: Callable,
    rate_limiter: Optional[RateLimiter],
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Process a single item and return result.
    
    Returns:
        (item_id, result_dict, error_string)
    """
    iid = get_item_id_fn(item)
    typ = get_item_type_fn(item)
    img_path = resolve_image_path_fn(item, images_root, jsonl_path)
    prompt = build_prompt_fn(item)
    
    # Acquire rate limit token
    if rate_limiter:
        rate_limiter.acquire()
    
    t0 = time.time()
    err: Optional[str] = None
    txt = ""
    
    try:
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        txt = call_model_fn(spec, img_path, prompt, max_output_tokens=max_output_tokens)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    
    dt = time.time() - t0
    
    return (
        iid,
        {
            "provider": spec.provider,
            "model": spec.model,
            "id": iid,
            "type": typ,
            "image_path": str(img_path),
            "prompt": prompt,
            "raw_text": txt,
            "parsed_json": None,  # Will be parsed later
            "latency_s": dt,
            "error": err,
        },
        err,
    )


def run_concurrent_evaluation(
    items: List[Dict[str, Any]],
    spec: Any,  # ModelSpec
    images_root: Optional[Path],
    jsonl_path: Path,
    max_output_tokens: int,
    call_model_fn: Callable,
    build_prompt_fn: Callable,
    resolve_image_path_fn: Callable,
    get_item_id_fn: Callable,
    get_item_type_fn: Callable,
    extract_first_json_fn: Callable,
    score_item_fields_fn: Callable,
    policy: str,
    raw_file_path: Path,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run evaluation with concurrent requests and rate limiting.
    
    Args:
        items: List of items to evaluate
        spec: ModelSpec for the model
        max_workers: Number of concurrent workers (None = use provider default)
        ... other args for processing
    
    Returns:
        List of scored rows
    """
    # Get rate limit config for provider
    rate_limit = PROVIDER_RATE_LIMITS.get(spec.provider)
    if rate_limit is None:
        # Default conservative limits
        rate_limit = RateLimit(requests_per_minute=30, max_concurrent=5)
    
    # Determine number of workers
    if max_workers is None:
        max_workers = rate_limit.max_concurrent
    else:
        max_workers = min(max_workers, rate_limit.max_concurrent)
    
    # Create rate limiter
    rate_limiter = RateLimiter(rate_limit.requests_per_minute) if rate_limit.requests_per_minute > 0 else None
    
    print(f"[concurrent] Using {max_workers} workers, {rate_limit.requests_per_minute} req/min limit for {spec.provider}")
    
    all_rows: List[Dict[str, Any]] = []
    completed_count = 0
    
    # Open raw file for writing
    with raw_file_path.open("w", encoding="utf-8") as raw_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    process_item,
                    item=it,
                    spec=spec,
                    images_root=images_root,
                    jsonl_path=jsonl_path,
                    max_output_tokens=max_output_tokens,
                    call_model_fn=call_model_fn,
                    build_prompt_fn=build_prompt_fn,
                    resolve_image_path_fn=resolve_image_path_fn,
                    get_item_id_fn=get_item_id_fn,
                    get_item_type_fn=get_item_type_fn,
                    rate_limiter=rate_limiter,
                ): it
                for it in items
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(items), desc=f"{spec.provider}:{spec.model}") as pbar:
                for future in as_completed(futures):
                    try:
                        iid, result_dict, err = future.result()
                        
                        # Parse JSON
                        parsed = extract_first_json_fn(result_dict["raw_text"]) if result_dict["raw_text"] else None
                        result_dict["parsed_json"] = parsed
                        
                        # Score fields
                        item = futures[future]
                        rows = score_item_fields_fn(
                            it=item,
                            parsed_json=parsed,
                            provider=spec.provider,
                            model=spec.model,
                            policy=policy,
                            latency_s=result_dict["latency_s"],
                            error=err,
                        )
                        all_rows.extend(rows)
                        
                        # Write to raw file (thread-safe since we're writing sequentially)
                        raw_f.write(
                            json.dumps(result_dict, ensure_ascii=False) + "\n"
                        )
                        raw_f.flush()  # Ensure data is written
                        
                        completed_count += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        pbar.write(f"[error] Failed to process item: {e}")
                        completed_count += 1
                        pbar.update(1)
    
    return all_rows

