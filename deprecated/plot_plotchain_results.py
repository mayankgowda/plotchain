#!/usr/bin/env python3
"""
plot_plotchain_results.py

Reads results_v2/*.csv from run_plotchain_eval.py and produces:
- pass_rate_by_type.png
- mae_by_type.png
- leaderboard.md
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results_v2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    per_item = pd.read_csv(out_dir / "per_item.csv")
    overall = pd.read_csv(out_dir / "overall.csv")

    # Leaderboard markdown
    md = []
    md.append("# PlotChain v2 — Leaderboard\n")
    md.append(overall.to_markdown(index=False))
    (out_dir / "leaderboard.md").write_text("\n".join(md), encoding="utf-8")

    # Pass rate by type (averaged over fields)
    by_type = (
        per_item.groupby(["provider", "model", "type"])
        .agg(pass_rate=("pass", "mean"), mean_abs_err=("abs_err", "mean"))
        .reset_index()
    )

    # PASS plot
    plt.figure()
    for (provider, model), g in by_type.groupby(["provider", "model"]):
        plt.plot(g["type"], g["pass_rate"], marker="o", label=f"{provider}:{model}")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Pass rate")
    plt.title("PlotChain v2 — Pass rate by plot type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pass_rate_by_type.png", dpi=200)

    # MAE plot
    plt.figure()
    for (provider, model), g in by_type.groupby(["provider", "model"]):
        plt.plot(g["type"], g["mean_abs_err"], marker="o", label=f"{provider}:{model}")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean absolute error")
    plt.title("PlotChain v2 — MAE by plot type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mae_by_type.png", dpi=200)

    print(f"Wrote:\n- {out_dir/'leaderboard.md'}\n- {out_dir/'pass_rate_by_type.png'}\n- {out_dir/'mae_by_type.png'}")


if __name__ == "__main__":
    main()
