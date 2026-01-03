#!/usr/bin/env python3
# generate_plotchain_v3.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common import ensure_dir, write_jsonl
from families import REGISTRY


def analyze_dataset(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for it in items:
        gt = it.get("ground_truth", {}) or {}
        gen = it.get("generation", {}) or {}
        rows.append({
            "id": it.get("id", ""),
            "type": it.get("type", ""),
            "difficulty": gen.get("difficulty", ""),
            "edge_tag": gen.get("edge_tag", ""),
            "n_gt_fields": len(gt),
        })
    return pd.DataFrame(rows)


def field_stats(items: List[Dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for it in items:
        typ = it.get("type", "")
        gt = it.get("ground_truth", {}) or {}
        for k, v in gt.items():
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                flat.append({"type": typ, "field": k, "value": float(v)})
    df = pd.DataFrame(flat)
    if df.empty:
        return df
    return (
        df.groupby(["type", "field"])
          .agg(n=("value", "size"),
               mean=("value", "mean"),
               std=("value", "std"),
               min=("value", "min"),
               max=("value", "max"))
          .reset_index()
          .sort_values(["type", "field"])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/plotchain_v3", help="Output directory")
    ap.add_argument("--seed", type=int, default=0, help="Master seed (deterministic)")
    ap.add_argument("--n_per_type", type=int, default=15, help="Items per plot family")
    ap.add_argument("--types", type=str, nargs="*", default=[], help="Optional subset of types to generate")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    images_root = out_dir / "images"
    ensure_dir(out_dir)
    ensure_dir(images_root)

    chosen_types = args.types if args.types else list(REGISTRY.keys())
    for t in chosen_types:
        if t not in REGISTRY:
            raise SystemExit(f"Unknown type: {t}. Known: {sorted(REGISTRY.keys())}")

    jsonl_path = out_dir / "plotchain_v3.jsonl"
    validation_csv = out_dir / "validation_report.csv"
    analysis_csv = out_dir / "dataset_analysis.csv"
    fieldstats_csv = out_dir / "field_stats.csv"

    if (jsonl_path.exists() or validation_csv.exists()) and not args.overwrite:
        raise SystemExit(f"Output exists in {out_dir}. Use --overwrite to regenerate.")

    all_items: List[Dict[str, Any]] = []
    all_validation_rows: List[Dict[str, Any]] = []

    for typ in chosen_types:
        mod = REGISTRY[typ]
        n = int(args.n_per_type) if args.n_per_type > 0 else int(getattr(mod, "DEFAULT_N", 15))
        print(f"[gen] {typ}: n={n}")
        items = mod.generate(out_dir=out_dir, master_seed=args.seed, n=n, images_root=images_root)
        all_items.extend(items)
        all_validation_rows.extend(mod.validate(items))

    write_jsonl(jsonl_path, all_items)
    print(f"[write] {jsonl_path} ({len(all_items)} items)")

    vdf = pd.DataFrame(all_validation_rows)
    if not vdf.empty:
        vdf.to_csv(validation_csv, index=False)
        pass_rate = float(vdf["pass"].mean()) if "pass" in vdf.columns else float("nan")
        print(f"[validate] pass_rate={pass_rate*100:.1f}%  rows={len(vdf)}")

        summary = (
            vdf.groupby(["type", "field"])
               .agg(n=("pass", "size"),
                    pass_rate=("pass", "mean"),
                    max_abs_err=("abs_err", "max"))
               .reset_index()
               .sort_values(["type", "field"])
        )
        print("\n=== Baseline Validation Summary ===")
        print(summary.to_string(index=False))
    else:
        print("[validate] no validation rows produced?")

    adf = analyze_dataset(all_items)
    if not adf.empty:
        adf.to_csv(analysis_csv, index=False)
        counts = adf.groupby(["type", "difficulty"]).size().reset_index(name="n")
        print("\n=== Counts by type/difficulty ===")
        print(counts.to_string(index=False))
        print(f"[analysis] wrote {analysis_csv}")

    fs = field_stats(all_items)
    if not fs.empty:
        fs.to_csv(fieldstats_csv, index=False)
        print(f"[stats] wrote {fieldstats_csv}")


if __name__ == "__main__":
    main()
