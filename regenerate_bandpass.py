#!/usr/bin/env python3
"""
Selectively regenerate only bandpass_response plots and update the dataset.
"""

import json
from pathlib import Path
from generate_plotchain_v4 import generate_family, validate_items

def regenerate_bandpass_only(out_dir: Path, master_seed: int = 0, n_per_family: int = 30):
    """Regenerate only bandpass_response items and update the JSONL file."""
    
    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    
    # Load existing JSONL
    jsonl_path = out_dir / "plotchain_v4.jsonl"
    existing_items = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_items.append(json.loads(line))
        print(f"[load] Loaded {len(existing_items)} existing items")
    
    # Filter out old bandpass items
    non_bandpass_items = [it for it in existing_items if it["type"] != "bandpass_response"]
    print(f"[filter] Keeping {len(non_bandpass_items)} non-bandpass items")
    
    # Generate new bandpass items
    print(f"[gen] Generating bandpass_response items...")
    new_bandpass_items = generate_family(
        "bandpass_response",
        out_dir=out_dir,
        images_root=images_root,
        master_seed=master_seed,
        n=n_per_family
    )
    print(f"[gen] Generated {len(new_bandpass_items)} bandpass_response items")
    
    # Combine: non-bandpass + new bandpass
    all_items = non_bandpass_items + new_bandpass_items
    
    # Sort by type and id to maintain consistent ordering
    all_items.sort(key=lambda x: (x["type"], x["id"]))
    
    # Write updated JSONL
    with jsonl_path.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[write] Updated {jsonl_path} ({len(all_items)} items)")
    
    # Validate
    validate_items(all_items, out_dir)
    
    return all_items

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/plotchain_v4", help="Output directory")
    ap.add_argument("--n_per_family", type=int, default=30, help="Items per plot family")
    ap.add_argument("--seed", type=int, default=0, help="Master seed")
    args = ap.parse_args()
    
    regenerate_bandpass_only(
        out_dir=Path(args.out_dir),
        master_seed=args.seed,
        n_per_family=args.n_per_family
    )

