#!/usr/bin/env python3
"""
Selectively re-run evaluations for only bandpass_response items.
This updates existing result files by replacing only bandpass_response rows.
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Import from the main eval script
sys.path.insert(0, str(Path(__file__).parent))
from run_plotchain_v4_eval import (
    load_jsonl, parse_models, call_model, build_prompt, resolve_image_path,
    get_item_id, get_item_type, extract_first_json, score_item_fields, write_reports
)

def eval_bandpass_only(
    jsonl_path: Path,
    images_root: Path,
    specs,  # List[ModelSpec]
    out_dir: Path,
    policy: str,
    max_output_tokens: int,
    overwrite: bool,
    concurrent: int = None,
    sequential: bool = False,
):
    """Re-run evaluation for only bandpass_response items."""
    
    # Load all items
    all_items = load_jsonl(jsonl_path)
    
    # Filter to only bandpass_response
    bandpass_items = [it for it in all_items if it["type"] == "bandpass_response"]
    print(f"[filter] Found {len(bandpass_items)} bandpass_response items")
    
    if len(bandpass_items) == 0:
        print("[error] No bandpass_response items found!")
        return
    
    # Process each model
    all_new_rows = []
    
    for spec in specs:
        base_name = f"raw_{spec.provider}_{spec.model.replace('/','_')}.jsonl"
        raw_path = out_dir / base_name
        
        # Load existing raw outputs
        existing_raw = []
        if raw_path.exists():
            with raw_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_raw.append(json.loads(line))
            print(f"[load] Loaded {len(existing_raw)} existing raw outputs for {spec.provider}:{spec.model}")
        
        # Filter out old bandpass items
        non_bandpass_raw = [r for r in existing_raw if r.get("type") != "bandpass_response"]
        print(f"[filter] Keeping {len(non_bandpass_raw)} non-bandpass raw outputs")
        
        # Evaluate bandpass items
        print(f"[eval] Evaluating {len(bandpass_items)} bandpass_response items for {spec.provider}:{spec.model}...")
        
        # Try concurrent evaluation
        bandpass_rows = []
        bandpass_raw_new = []
        
        try:
            from concurrent_eval import run_concurrent_evaluation
            if not sequential:
                print(f"[concurrent] Using concurrent evaluation...")
                # Create temp raw file for bandpass only
                temp_raw_path = out_dir / f"raw_{spec.provider}_{spec.model.replace('/','_')}_bandpass_temp.jsonl"
                
                bandpass_rows = run_concurrent_evaluation(
                    items=bandpass_items,
                    spec=spec,
                    images_root=images_root,
                    jsonl_path=jsonl_path,
                    max_output_tokens=max_output_tokens,
                    call_model_fn=call_model,
                    build_prompt_fn=build_prompt,
                    resolve_image_path_fn=resolve_image_path,
                    get_item_id_fn=get_item_id,
                    get_item_type_fn=get_item_type,
                    extract_first_json_fn=extract_first_json,
                    score_item_fields_fn=score_item_fields,
                    policy=policy,
                    raw_file_path=temp_raw_path,
                    max_workers=concurrent,
                )
                
                # Load the temp raw file
                with temp_raw_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            bandpass_raw_new.append(json.loads(line))
                
                # Clean up temp file
                temp_raw_path.unlink()
                
        except Exception as e:
            print(f"[warning] Concurrent evaluation failed: {e}, using sequential")
            sequential = True
        
        # Sequential fallback
        if sequential or len(bandpass_rows) == 0:
            print(f"[sequential] Using sequential evaluation...")
            import time
            from tqdm import tqdm
            
            for it in tqdm(bandpass_items, desc=f"{spec.provider}:{spec.model}"):
                iid = get_item_id(it)
                typ = get_item_type(it)
                img_path = resolve_image_path(it, images_root, jsonl_path)
                prompt = build_prompt(it)
                
                t0 = time.time()
                err = None
                txt = ""
                try:
                    if not img_path.exists():
                        raise FileNotFoundError(f"Image not found: {img_path}")
                    txt = call_model(spec, img_path, prompt, max_output_tokens=max_output_tokens)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                dt = time.time() - t0
                
                parsed = extract_first_json(txt) if txt else None
                rows = score_item_fields(
                    it=it,
                    parsed_json=parsed,
                    provider=spec.provider,
                    model=spec.model,
                    policy=policy,
                    latency_s=dt,
                    error=err,
                )
                bandpass_rows.extend(rows)
                
                bandpass_raw_new.append({
                    "provider": spec.provider,
                    "model": spec.model,
                    "id": iid,
                    "type": typ,
                    "image_path": str(img_path),
                    "prompt": prompt,
                    "raw_text": txt,
                    "parsed_json": parsed,
                    "latency_s": dt,
                    "error": err,
                })
        
        all_new_rows.extend(bandpass_rows)
        
        # Merge raw outputs: non-bandpass + new bandpass
        merged_raw = non_bandpass_raw + bandpass_raw_new
        
        # Write merged raw file
        if overwrite:
            with raw_path.open("w", encoding="utf-8") as f:
                for r in merged_raw:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[write] Updated {raw_path} ({len(merged_raw)} total)")
    
    if not all_new_rows:
        print("[error] No rows produced")
        return
    
    # Update per_item.csv
    per_item_csv = out_dir / "per_item.csv"
    if per_item_csv.exists():
        existing_df = pd.read_csv(per_item_csv)
        # Remove old bandpass rows
        existing_df = existing_df[existing_df["type"] != "bandpass_response"]
        print(f"[filter] Kept {len(existing_df)} non-bandpass rows from existing results")
        
        # Add new bandpass rows
        new_df = pd.DataFrame(all_new_rows)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_csv(per_item_csv, index=False)
        print(f"[write] Updated {per_item_csv} ({len(updated_df)} total rows)")
    else:
        # Create new file
        df = pd.DataFrame(all_new_rows)
        df.to_csv(per_item_csv, index=False)
        print(f"[write] Created {per_item_csv}")
    
    # Re-score to regenerate all reports
    print(f"[score] Re-generating all reports...")
    df = pd.read_csv(per_item_csv)
    write_reports(df, out_dir)
    print(f"[done] Evaluation complete!")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to plotchain_v4.jsonl")
    ap.add_argument("--images_root", type=str, default="", help="Root directory for images")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory (e.g., results/gpt41_plotread)")
    ap.add_argument("--models", type=str, nargs="*", required=True, help="Models to evaluate (e.g., openai:gpt-4.1)")
    ap.add_argument("--policy", type=str, default="plotread", choices=["plotread", "strict"])
    ap.add_argument("--max_output_tokens", type=int, default=400)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    ap.add_argument("--concurrent", type=int, default=None)
    ap.add_argument("--sequential", action="store_true")
    
    args = ap.parse_args()
    
    specs = parse_models(args.models)
    images_root = Path(args.images_root).resolve() if args.images_root else None
    
    eval_bandpass_only(
        jsonl_path=Path(args.jsonl),
        images_root=images_root,
        specs=specs,
        out_dir=Path(args.out_dir),
        policy=args.policy,
        max_output_tokens=args.max_output_tokens,
        overwrite=args.overwrite,
        concurrent=args.concurrent,
        sequential=args.sequential,
    )
