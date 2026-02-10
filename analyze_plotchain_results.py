#!/usr/bin/env python3
"""
PlotChain Results Analysis Script

This script analyzes evaluation results from multiple model runs and generates
publication-ready outputs for IEEE conference paper.

Usage:
    python3 analyze_plotchain_results.py --runs_dir results --output_dir analysis_output

Inputs:
    Expects subdirectories under --runs_dir, each containing:
    - item_level.csv (required)
    - per_item.csv (required)
    - overall.csv (optional)
    - summary.csv (optional)
    - raw_*.jsonl (optional, for error analysis)

Outputs:
    - paper_numbers.json: Definitive metrics for paper citation
    - tables/: LaTeX tables (.tex) and CSV versions
    - figures/: PDF figures for paper inclusion
    - statistics/: Paired comparison statistics (CSV + LaTeX)

Author: PlotChain Analysis Pipeline
Date: January 2026
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# For figures
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from scipy import stats
from scipy.stats import ttest_rel
try:
    from scipy.stats.contingency import mcnemar
except ImportError:
    # Fallback: use statsmodels or manual implementation
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        # Manual McNemar implementation
        def mcnemar(table, exact=False, correction=True):
            """Manual McNemar test implementation."""
            from scipy.stats import chi2
            both_pass, model1_only = table[0]
            model2_only, both_fail = table[1]
            
            if exact:
                # Exact test (not implemented here)
                statistic = np.nan
                pvalue = 1.0
            else:
                # Chi-square approximation
                b = model1_only
                c = model2_only
                if correction:
                    statistic = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
                else:
                    statistic = (b - c)**2 / (b + c) if (b + c) > 0 else 0
                pvalue = 1 - chi2.cdf(statistic, df=1) if (b + c) > 0 else 1.0
            
            class Result:
                def __init__(self, stat, pval):
                    self.statistic = stat
                    self.pvalue = pval
            return Result(statistic, pvalue)

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    # Fallback: simple Bonferroni correction
    def multipletests(pvals, method='holm', alpha=0.05):
        n = len(pvals)
        sorted_indices = np.argsort(pvals)
        p_corrected = np.zeros(n)
        for i, idx in enumerate(sorted_indices):
            p_corrected[idx] = min(pvals[idx] * (n - i), 1.0)
        return np.array([True] * n), p_corrected, alpha, None

# Set style for IEEE papers
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('default')
if HAS_SEABORN:
    sns.set_palette("husl")

# Deterministic random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Model name mapping: folder name -> display name
MODEL_NAME_MAP = {
    "gpt41_plotread_20260104_temp0": "GPT-4.1",
    "claudesonnet45_plotread__temp0": "Claude Sonnet 4.5",
    "gemini25pro_plotread_temp0_8192tokens": "Gemini 2.5 Pro",
    "gpt4o_plotread_20260104_temp0": "GPT-4o",
}

MODEL_VERSIONS = {
    "GPT-4.1": "gpt-4.1",
    "Claude Sonnet 4.5": "claude-sonnet-4-5-20250929",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "GPT-4o": "gpt-4o",
}

def get_display_name(folder_name: str) -> str:
    """Convert folder name to clean display name."""
    if folder_name in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[folder_name]
    
    # Try to infer from folder name
    folder_lower = folder_name.lower()
    if "gpt41" in folder_lower or "gpt-4.1" in folder_lower:
        return "GPT-4.1"
    elif "claude" in folder_lower:
        return "Claude Sonnet 4.5"
    elif "gemini" in folder_lower:
        return "Gemini 2.5 Pro"
    elif "gpt4o" in folder_lower or "gpt-4o" in folder_lower:
        return "GPT-4o"
    else:
        # Fallback: clean up folder name
        return folder_name.replace("_", " ").title()

def get_short_model_name(model_name: str) -> str:
    """Get short model name for tables/figures."""
    if "GPT-4.1" in model_name:
        return "GPT-4.1"
    elif "GPT-4o" in model_name:
        return "GPT-4o"
    elif "Claude" in model_name:
        return "Claude 4.5"
    elif "Gemini" in model_name:
        return "Gemini 2.5"
    return model_name.replace("_", "\\_")


def discover_model_runs(runs_dir: Path) -> Dict[str, Path]:
    """Discover all model run directories."""
    runs = {}
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    for subdir in sorted(runs_dir.iterdir()):
        if subdir.is_dir():
            item_level = subdir / "item_level.csv"
            per_item = subdir / "per_item.csv"
            if item_level.exists() and per_item.exists():
                display_name = get_display_name(subdir.name)
                runs[display_name] = subdir
            else:
                print(f"⚠️  Skipping {subdir.name}: missing required CSV files")
    
    return runs


def load_model_data(model_dir: Path, model_name: str) -> Dict:
    """Load all data files for a model."""
    data = {"name": model_name, "dir": model_dir}
    
    # Required files
    try:
        data["item_level"] = pd.read_csv(model_dir / "item_level.csv")
        data["per_item"] = pd.read_csv(model_dir / "per_item.csv")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load required files for {model_name}: {e}")
    
    # Optional files
    if (model_dir / "overall.csv").exists():
        data["overall"] = pd.read_csv(model_dir / "overall.csv").iloc[0].to_dict()
    
    if (model_dir / "summary.csv").exists():
        data["summary"] = pd.read_csv(model_dir / "summary.csv")
    
    # Find raw JSONL files for error analysis
    raw_files = list(model_dir.glob("raw_*.jsonl"))
    if raw_files:
        data["raw_file"] = raw_files[0]
    
    return data


def validate_item_ids(all_data: Dict[str, Dict]) -> List[str]:
    """Validate that all models share the same item IDs."""
    item_ids_sets = {}
    for model_name, data in all_data.items():
        if "item_level" in data:
            item_ids_sets[model_name] = set(data["item_level"]["id"].unique())
        elif "per_item" in data:
            item_ids_sets[model_name] = set(data["per_item"]["id"].unique())
    
    if not item_ids_sets:
        raise ValueError("No item IDs found in any model")
    
    # Find intersection
    common_ids = set.intersection(*item_ids_sets.values())
    
    # Check for differences
    for model_name, ids in item_ids_sets.items():
        missing = ids - common_ids
        extra = common_ids - ids
        if missing:
            print(f"⚠️  {model_name}: {len(missing)} items not in common set")
        if extra:
            print(f"⚠️  {model_name}: {len(extra)} items missing from this model")
    
    print(f"✅ Common item set: {len(common_ids)} items")
    return sorted(list(common_ids))


def compute_paper_numbers(all_data: Dict[str, Dict], common_ids: List[str]) -> Dict:
    """Compute all key metrics for paper."""
    numbers = {
        "models": {},
        "overall_leaderboard": [],
        "family_performance": {},
        "paired_comparisons": [],
    }
    
    # Overall metrics per model
    for model_name, data in all_data.items():
        model_nums = {}
        
        # A) Field-level metrics (from per_item.csv)
        if "per_item" in data:
            per_item = data["per_item"]
            per_item_common = per_item[per_item["id"].isin(common_ids)].copy()
            
            # Fix NaN handling: treat NaN as failure (0)
            if "pass" in per_item_common.columns:
                per_item_common["pass"] = per_item_common["pass"].fillna(0).astype(float)
            
            # Overall field pass rate
            if "pass" in per_item_common.columns:
                model_nums["overall_field_pass_rate"] = float(per_item_common["pass"].mean())
            
            # Final field pass rate
            if "is_checkpoint" in per_item_common.columns:
                final_fields = per_item_common[~per_item_common["is_checkpoint"]]
                checkpoint_fields = per_item_common[per_item_common["is_checkpoint"]]
                
                if "pass" in final_fields.columns:
                    model_nums["final_field_pass_rate"] = float(final_fields["pass"].mean())
                if "pass" in checkpoint_fields.columns:
                    model_nums["checkpoint_field_pass_rate"] = float(checkpoint_fields["pass"].mean())
            
            # Error rates
            if "error" in per_item_common.columns:
                error_count = per_item_common["error"].notna().sum()
                model_nums["error_rate"] = float(error_count / len(per_item_common))
            
            # Latency
            if "latency_s" in per_item_common.columns:
                model_nums["mean_latency_s"] = float(per_item_common["latency_s"].mean())
            
            # B) Item-level metrics (headline)
            # Item final all-pass rate: ALL final fields must pass
            if "is_checkpoint" in per_item_common.columns and "pass" in per_item_common.columns:
                final_fields = per_item_common[~per_item_common["is_checkpoint"]]
                # Group by id and check if ALL final fields pass
                item_final_allpass = final_fields.groupby("id")["pass"].all()
                model_nums["item_final_allpass_rate"] = float(item_final_allpass.mean())
                
                # Item checkpoint all-pass rate: ALL checkpoint fields must pass
                checkpoint_fields = per_item_common[per_item_common["is_checkpoint"]]
                item_checkpoint_allpass = checkpoint_fields.groupby("id")["pass"].all()
                model_nums["item_checkpoint_allpass_rate"] = float(item_checkpoint_allpass.mean())
                
                # Item final field accuracy: mean over items of mean(pass over final fields)
                item_final_accuracy = final_fields.groupby("id")["pass"].mean()
                model_nums["item_final_field_accuracy"] = float(item_final_accuracy.mean())
        
        # Fallback: compute from item_level if per_item not available
        elif "item_level" in data:
            item_level = data["item_level"]
            item_level_common = item_level[item_level["id"].isin(common_ids)].copy()
            
            # Fix NaN handling
            if "pass" in item_level_common.columns:
                item_level_common["pass"] = item_level_common["pass"].fillna(0).astype(float)
            
            # Field-level metrics
            if "pass" in item_level_common.columns:
                model_nums["overall_field_pass_rate"] = float(item_level_common["pass"].mean())
            
            if "is_checkpoint" in item_level_common.columns:
                final = item_level_common[~item_level_common["is_checkpoint"]]
                checkpoint = item_level_common[item_level_common["is_checkpoint"]]
                if "pass" in final.columns:
                    model_nums["final_field_pass_rate"] = float(final["pass"].mean())
                if "pass" in checkpoint.columns:
                    model_nums["checkpoint_field_pass_rate"] = float(checkpoint["pass"].mean())
        
        # Additional metrics from overall.csv if available
        if "overall" in data:
            overall = data["overall"]
            model_nums["mean_abs_err"] = float(overall.get("mean_abs_err", 0))
            model_nums["mean_rel_err"] = float(overall.get("mean_rel_err", 0))
            model_nums["total_fields"] = int(overall.get("n", 0))
            # Use overall.csv latency if per_item didn't have it
            if "mean_latency_s" not in model_nums:
                model_nums["mean_latency_s"] = float(overall.get("mean_latency_s", 0))
        
        numbers["models"][model_name] = model_nums
    
    # Leaderboard ranking - use item_final_allpass_rate as headline metric
    leaderboard = []
    for model_name, model_nums in numbers["models"].items():
        leaderboard.append({
            "model": model_name,
            # Field-level metrics
            "overall_field_pass_rate": model_nums.get("overall_field_pass_rate", 0),
            "final_field_pass_rate": model_nums.get("final_field_pass_rate", 0),
            "checkpoint_field_pass_rate": model_nums.get("checkpoint_field_pass_rate", 0),
            # Item-level metrics (headline)
            "item_final_allpass_rate": model_nums.get("item_final_allpass_rate", 0),
            "item_checkpoint_allpass_rate": model_nums.get("item_checkpoint_allpass_rate", 0),
            "item_final_field_accuracy": model_nums.get("item_final_field_accuracy", 0),
            # Other metrics
            "mean_latency_s": model_nums.get("mean_latency_s", 0),
        })
    # Sort by item_final_allpass_rate (headline metric)
    leaderboard.sort(key=lambda x: x["item_final_allpass_rate"], reverse=True)
    numbers["overall_leaderboard"] = leaderboard
    
    # Family performance - compute field-level final pass rate per family (for heatmap visualization)
    # Note: Item-level all-pass is stored separately in model metrics
    all_families = set()
    for data in all_data.values():
        if "per_item" in data:
            all_families.update(data["per_item"]["type"].unique())
        elif "summary" in data:
            all_families.update(data["summary"]["type"].unique())
    
    for family in sorted(all_families):
        family_nums = {}
        for model_name, data in all_data.items():
            if "per_item" in data:
                per_item = data["per_item"]
                per_item_common = per_item[per_item["id"].isin(common_ids)].copy()
                family_items = per_item_common[per_item_common["type"] == family]
                
                if len(family_items) > 0:
                    # Fix NaN handling
                    if "pass" in family_items.columns:
                        family_items = family_items.copy()  # Avoid SettingWithCopyWarning
                        family_items.loc[:, "pass"] = family_items["pass"].fillna(0).astype(float)
                    
                    # Use field-level final pass rate for heatmap (more informative than item-level all-pass)
                    if "is_checkpoint" in family_items.columns:
                        final_fields = family_items[~family_items["is_checkpoint"]]
                        if "pass" in final_fields.columns and len(final_fields) > 0:
                            # Field-level pass rate (mean pass over all final fields)
                            family_nums[model_name] = float(final_fields["pass"].mean())
                    else:
                        # Fallback to overall field-level pass rate
                        if "pass" in family_items.columns:
                            family_nums[model_name] = float(family_items["pass"].mean())
            elif "summary" in data:
                # Fallback to summary if per_item not available
                family_data = data["summary"][data["summary"]["type"] == family]
                if len(family_data) > 0:
                    family_nums[model_name] = float(family_data["pass_rate"].mean())
        if family_nums:
            numbers["family_performance"][family] = family_nums
    
    return numbers


def generate_overall_leaderboard_table(all_data: Dict[str, Dict], numbers: Dict, output_dir: Path):
    """Generate overall leaderboard table."""
    leaderboard = numbers["overall_leaderboard"]
    
    # CSV - include both field-level and item-level metrics
    df = pd.DataFrame(leaderboard)
    df["rank"] = range(1, len(df) + 1)
    df = df[["rank", "model", 
             "overall_field_pass_rate", "final_field_pass_rate", "checkpoint_field_pass_rate",
             "item_final_allpass_rate", "item_checkpoint_allpass_rate", "item_final_field_accuracy",
             "mean_latency_s"]]
    df.columns = ["Rank", "Model", 
                  "Overall Field Pass Rate", "Final-Field Pass Rate", "Checkpoint-Field Pass Rate",
                  "Item Final All-Pass Rate", "Item Checkpoint All-Pass Rate", "Item Final Field Accuracy",
                  "Mean Latency (s)"]
    df.to_csv(output_dir / "tables" / "overall_leaderboard.csv", index=False)
    
    # LaTeX - include both metric types with clear labels
    latex = "\\begin{table}[h]\n\\centering\n\\caption{Overall Performance Leaderboard}\n"
    latex += "\\label{tab:leaderboard}\n\\footnotesize\n"
    latex += "\\begin{tabular}{lccccccc}\n\\toprule\n"
    latex += "Model & \\multicolumn{3}{c}{Field-Level Pass Rates} & \\multicolumn{3}{c}{Item-Level Metrics} & Latency \\\\\n"
    latex += "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
    latex += " & Overall & Final & Checkpoint & Final All-Pass & Checkpoint All-Pass & Final Accuracy & (s) \\\\\n\\midrule\n"
    
    for entry in leaderboard:
        model = get_short_model_name(entry["model"])
        latex += f"{model} & "
        # Field-level
        latex += f"{entry.get('overall_field_pass_rate', 0)*100:.2f}\\% & "
        latex += f"{entry.get('final_field_pass_rate', 0)*100:.2f}\\% & "
        latex += f"{entry.get('checkpoint_field_pass_rate', 0)*100:.2f}\\% & "
        # Item-level
        latex += f"{entry.get('item_final_allpass_rate', 0)*100:.2f}\\% & "
        latex += f"{entry.get('item_checkpoint_allpass_rate', 0)*100:.2f}\\% & "
        latex += f"{entry.get('item_final_field_accuracy', 0)*100:.2f}\\% & "
        # Latency
        latex += f"{entry.get('mean_latency_s', 0):.2f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open(output_dir / "tables" / "overall_leaderboard.tex", "w") as f:
        f.write(latex)


def generate_family_matrix_table(all_data: Dict[str, Dict], numbers: Dict, output_dir: Path):
    """Generate family-by-model pass rate matrix."""
    family_perf = numbers["family_performance"]
    models = sorted(all_data.keys())
    
    # Build DataFrame
    rows = []
    for family in sorted(family_perf.keys()):
        row = {"Family": family}
        for model in models:
            row[model] = family_perf[family].get(model, np.nan) * 100
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tables" / "family_matrix.csv", index=False)
    
    # LaTeX
    models_short = [get_short_model_name(m) for m in models]
    latex = "\\begin{table}[h]\n\\centering\n\\caption{Family-Level Performance Matrix (Final-Field Pass Rate \\%)}\n"
    latex += "\\label{tab:family_matrix}\n\\resizebox{\\textwidth}{!}{%\n"
    latex += "\\begin{tabular}{l" + "c" * len(models) + "}\n\\toprule\n"
    latex += "Family & " + " & ".join(models_short) + " \\\\\n\\midrule\n"
    
    for family in sorted(family_perf.keys()):
        latex += f"{family.replace('_', '\\_')} & "
        values = []
        for model in models:
            val = family_perf[family].get(model, np.nan)
            if not np.isnan(val):
                values.append(f"{val*100:.1f}\\%")
            else:
                values.append("N/A")
        latex += " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    
    with open(output_dir / "tables" / "family_matrix.tex", "w") as f:
        f.write(latex)


def generate_hardest_families_table(all_data: Dict[str, Dict], output_dir: Path):
    """Generate hardest families table (bottom 5 per model)."""
    rows = []
    
    for model_name, data in all_data.items():
        if "summary" in data:
            summary = data["summary"]
            family_stats = summary.groupby("type")["pass_rate"].mean().sort_values()
            
            for i, (family, pass_rate) in enumerate(family_stats.head(5).items()):
                rows.append({
                    "Model": model_name,
                    "Rank": i + 1,
                    "Family": family,
                    "Pass Rate": pass_rate * 100,
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tables" / "hardest_families.csv", index=False)
    
    # LaTeX
    latex = "\\begin{table}[h]\n\\centering\n\\caption{Hardest Families (Bottom 5 per Model)}\n"
    latex += "\\label{tab:hardest_families}\n\\begin{tabular}{llcc}\n\\toprule\n"
    latex += "Model & Rank & Family & Pass Rate (\\%) \\\\\n\\midrule\n"
    
    for _, row in df.iterrows():
        model_short = get_short_model_name(row['Model'])
        latex += f"{model_short} & {int(row['Rank'])} & "
        latex += f"{row['Family'].replace('_', '\\_')} & {row['Pass Rate']:.1f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open(output_dir / "tables" / "hardest_families.tex", "w") as f:
        f.write(latex)


def generate_worst_fields_table(all_data: Dict[str, Dict], output_dir: Path, n: int = 10):
    """Generate worst fields table (bottom N per model)."""
    rows = []
    
    for model_name, data in all_data.items():
        if "summary" in data:
            summary = data["summary"]
            worst = summary.nsmallest(n, "pass_rate")
            
            for _, row in worst.iterrows():
                rows.append({
                    "Model": model_name,
                    "Family": row["type"],
                    "Field": row["field"],
                    "Pass Rate": row["pass_rate"] * 100,
                    "Mean Abs Err": row.get("mean_abs_err", np.nan),
                    "Count": int(row.get("n", 0)),
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tables" / "worst_fields.csv", index=False)
    
    # LaTeX (abbreviated)
    latex = "\\begin{table}[h]\n\\centering\n\\caption{Worst Performing Fields (Bottom 10 per Model)}\n"
    latex += "\\label{tab:worst_fields}\n\\footnotesize\n\\begin{tabular}{lllcc}\n\\toprule\n"
    latex += "Model & Family & Field & Pass Rate (\\%) & Count \\\\\n\\midrule\n"
    
    for _, row in df.iterrows():
        model_short = get_short_model_name(row['Model'])
        latex += f"{model_short} & "
        latex += f"{row['Family'].replace('_', '\\_')} & "
        latex += f"{row['Field'].replace('_', '\\_')} & "
        latex += f"{row['Pass Rate']:.1f} & {int(row['Count'])} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open(output_dir / "tables" / "worst_fields.tex", "w") as f:
        f.write(latex)


def generate_error_table(all_data: Dict[str, Dict], common_ids: List[str], output_dir: Path):
    """Generate format/compliance errors table."""
    rows = []
    
    for model_name, data in all_data.items():
        error_items = set()
        total_items = 0
        
        # Count unique items with errors (not fields!)
        # From per_item - count unique items with errors
        if "per_item" in data:
            per_item = data["per_item"]
            per_item_common = per_item[per_item["id"].isin(common_ids)]
            total_items = len(per_item_common["id"].unique())  # Unique items, not rows
            
            if "error" in per_item_common.columns:
                error_items.update(per_item_common[per_item_common["error"].notna()]["id"].unique())
        
        # From raw JSONL - count items with empty/null responses or errors
        if "raw_file" in data and data["raw_file"].exists():
            import json
            try:
                with open(data["raw_file"]) as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            item_id = item.get("id", "")
                            if item_id in common_ids:
                                # Check for empty/null responses or API errors
                                raw_text = item.get("raw_text", "")
                                parsed_json = item.get("parsed_json")
                                error = item.get("error", "")
                                
                                if (not raw_text or raw_text.strip() == "" or 
                                    parsed_json is None or 
                                    (error and error.strip() and any(x in error for x in ["API", "429", "500", "timeout"]))):
                                    error_items.add(item_id)
            except:
                pass
        
        error_count = len(error_items)
        error_rate = error_count / total_items if total_items > 0 else 0
        
        rows.append({
            "Model": model_name,
            "Error Count": error_count,
            "Total Items": total_items,
            "Error Rate": error_rate * 100,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tables" / "error_analysis.csv", index=False)
    
    # LaTeX
    latex = "\\begin{table}[h]\n\\centering\n\\caption{Format and Compliance Errors}\n"
    latex += "\\label{tab:errors}\n\\begin{tabular}{lccc}\n\\toprule\n"
    latex += "Model & Error Count & Total Items & Error Rate (\\%) \\\\\n\\midrule\n"
    
    for _, row in df.iterrows():
        model_short = get_short_model_name(row['Model'])
        latex += f"{model_short} & "
        latex += f"{int(row['Error Count'])} & {int(row['Total Items'])} & "
        latex += f"{row['Error Rate']:.2f} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open(output_dir / "tables" / "error_analysis.tex", "w") as f:
        f.write(latex)


def generate_figures(all_data: Dict[str, Dict], numbers: Dict, output_dir: Path):
    """Generate all figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    models = sorted(all_data.keys())
    
    # Helper to get short model name for figures
    def get_model_label(model_name: str) -> str:
        """Get short model label for figures."""
        return get_short_model_name(model_name)
    
    # 1. Overall pass rate bar chart - use item_final_allpass_rate as headline
    fig, ax = plt.subplots(figsize=(8, 5))
    leaderboard = numbers["overall_leaderboard"]
    models_sorted = [e["model"] for e in leaderboard]
    models_labels = [get_model_label(m) for m in models_sorted]
    pass_rates = [e.get("item_final_allpass_rate", e.get("overall_field_pass_rate", 0)) * 100 for e in leaderboard]
    
    if HAS_SEABORN:
        colors = sns.color_palette("husl", len(models_sorted))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_sorted)))
    bars = ax.bar(models_labels, pass_rates, color=colors)
    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Item Final All-Pass Rate by Model", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, pass_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "overall_pass_rate.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Final vs checkpoint pass rate
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models_sorted))
    width = 0.35
    
    final_rates = [e.get("final_field_pass_rate", 0) * 100 for e in leaderboard]
    checkpoint_rates = [e.get("checkpoint_field_pass_rate", 0) * 100 for e in leaderboard]
    
    bars1 = ax.bar(x - width/2, final_rates, width, label='Final Fields', color='#2ecc71')
    bars2 = ax.bar(x + width/2, checkpoint_rates, width, label='Checkpoint Fields', color='#3498db')
    
    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Final-Field vs Checkpoint-Field Pass Rates", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "final_vs_checkpoint.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Family pass rate heatmap
    family_perf = numbers["family_performance"]
    families = sorted(family_perf.keys())
    
    heatmap_data = []
    for family in families:
        row = []
        for model in models_sorted:
            val = family_perf[family].get(model, np.nan)
            row.append(val * 100 if not np.isnan(val) else np.nan)
        heatmap_data.append(row)
    
    # Optimize figure size for paper: smaller boxes, larger text
    n_families = len(families)
    n_models = len(models_sorted)
    # Use smaller figure size with more compact cells
    fig_width = max(8, n_models * 0.8)  # Smaller width per model
    fig_height = max(10, n_families * 0.5)  # Smaller height per family
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    models_labels = [get_model_label(m) for m in models_sorted]
    ax.set_xticks(np.arange(len(models_sorted)))
    ax.set_yticks(np.arange(len(families)))
    ax.set_xticklabels(models_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(families, fontsize=11)
    
    # Add text annotations with larger font
    for i in range(len(families)):
        for j in range(len(models_sorted)):
            val = heatmap_data[i][j]
            if not np.isnan(val):
                # Use larger font size, all text in black
                text = ax.text(j, i, f'{val:.0f}%', ha="center", va="center", 
                              color="black", fontsize=12, fontweight='bold')
    
    ax.set_title("Family-Level Performance Heatmap (Final-Field Pass Rate %)", 
                fontsize=16, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pass Rate (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "family_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Accuracy vs latency scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for entry in leaderboard:
        model_label = get_model_label(entry["model"])
        accuracy = entry.get("item_final_allpass_rate", entry.get("overall_field_pass_rate", 0)) * 100
        ax.scatter(entry.get("mean_latency_s", 0), accuracy,
                  s=200, alpha=0.7, label=model_label)
        ax.annotate(model_label, 
                   (entry.get("mean_latency_s", 0), accuracy),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel("Mean Latency (s)", fontsize=12)
    ax.set_ylabel("Item Final All-Pass Rate (%)", fontsize=12)
    ax.set_title("Accuracy vs Latency Trade-off", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_vs_latency.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Boxplot of family pass rates per model
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box_data = []
    box_labels = []
    for model in models_sorted:
        model_rates = []
        for family in families:
            val = family_perf[family].get(model, np.nan)
            if not np.isnan(val):
                model_rates.append(val * 100)
        if model_rates:
            box_data.append(model_rates)
            box_labels.append(get_model_label(model))
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Pass Rate (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Distribution of Family Pass Rates by Model", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(fig_dir / "family_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def bootstrap_paired_difference(x, y, n_bootstrap=10000, alpha=0.05):
    """Bootstrap CI for paired difference."""
    n = len(x)
    diffs = x - y
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        bootstrap_means.append(diffs[indices].mean())
    
    bootstrap_means = np.array(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return ci_lower, ci_upper


def compute_paired_statistics(all_data: Dict[str, Dict], common_ids: List[str], output_dir: Path):
    """Compute paired statistics between models."""
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(exist_ok=True)
    
    models = sorted(all_data.keys())
    n_models = len(models)
    
    # Prepare data: item-level final pass
    final_pass_data = {}
    accuracy_data = {}
    
    for model_name, data in all_data.items():
        # Compute final_pass and accuracy from per_item.csv
        if "per_item" in data:
            per_item = data["per_item"]
            per_item_common = per_item[per_item["id"].isin(common_ids)].copy()
            
            # Fix NaN handling before groupby
            if "pass" in per_item_common.columns:
                per_item_common["pass"] = per_item_common["pass"].fillna(0).astype(float)
            
            # Final pass: all final fields must pass for an item to pass
            # Filter to final fields only (not checkpoints)
            if "is_checkpoint" in per_item_common.columns:
                per_item_final = per_item_common[~per_item_common["is_checkpoint"]]
            else:
                per_item_final = per_item_common
            
            if "pass" in per_item_final.columns:
                # Group by id and check if ALL final fields pass
                final_pass = per_item_final.groupby("id")["pass"].all()
                # Convert boolean to float (True=1.0, False=0.0)
                final_pass_data[model_name] = final_pass.astype(float)
                
                # Accuracy: mean pass rate over final fields per item
                accuracy = per_item_final.groupby("id")["pass"].mean()
                accuracy_data[model_name] = accuracy.astype(float)  # Already handled NaN above
    
    # Align all dataframes
    if final_pass_data:
        final_pass_df = pd.DataFrame(final_pass_data)
        final_pass_df = final_pass_df.fillna(0)  # Missing = failure
    else:
        final_pass_df = None
    
    if accuracy_data:
        accuracy_df = pd.DataFrame(accuracy_data)
        accuracy_df = accuracy_df.fillna(0)  # Missing = failure
    else:
        accuracy_df = None
    
    # Compute pairwise comparisons
    comparisons = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1 = models[i]
            model2 = models[j]
            
            comp = {"model1": model1, "model2": model2}
            
            # Binary paired test (McNemar)
            if final_pass_df is not None and model1 in final_pass_df.columns and model2 in final_pass_df.columns:
                x = final_pass_df[model1].values
                y = final_pass_df[model2].values
                
                # Contingency table
                both_pass = ((x == 1) & (y == 1)).sum()
                model1_only = ((x == 1) & (y == 0)).sum()
                model2_only = ((x == 0) & (y == 1)).sum()
                both_fail = ((x == 0) & (y == 0)).sum()
                
                # McNemar test
                if model1_only + model2_only > 0:
                    try:
                        # Use scipy.stats.mcnemar directly
                        table = [[both_pass, model1_only], [model2_only, both_fail]]
                        result = mcnemar(table, exact=False, correction=True)
                        comp["mcnemar_statistic"] = float(result.statistic)
                        comp["mcnemar_pvalue"] = float(result.pvalue)
                    except Exception as e:
                        comp["mcnemar_statistic"] = np.nan
                        comp["mcnemar_pvalue"] = np.nan
                else:
                    comp["mcnemar_statistic"] = np.nan
                    comp["mcnemar_pvalue"] = 1.0
                
                # Bootstrap CI for difference
                mean_diff = x.mean() - y.mean()
                ci_lower, ci_upper = bootstrap_paired_difference(x, y)
                comp["final_pass_mean_diff"] = float(mean_diff)
                comp["final_pass_ci_lower"] = float(ci_lower)
                comp["final_pass_ci_upper"] = float(ci_upper)
                comp["final_pass_effect_size"] = float(mean_diff / np.sqrt(x.var() + y.var())) if (x.var() + y.var()) > 0 else np.nan
            
            # Continuous paired test (accuracy)
            if accuracy_df is not None and model1 in accuracy_df.columns and model2 in accuracy_df.columns:
                x = accuracy_df[model1].fillna(0.0).values  # Missing = failure (0.0)
                y = accuracy_df[model2].fillna(0.0).values  # Missing = failure (0.0)
                
                # All pairs should be valid now (no NaN after fillna)
                if len(x) > 1:
                    # Paired t-test
                    t_stat, p_value = ttest_rel(x, y)
                    comp["accuracy_t_statistic"] = float(t_stat)
                    comp["accuracy_pvalue"] = float(p_value)
                    
                    # Mean difference and CI
                    mean_diff = x.mean() - y.mean()
                    ci_lower, ci_upper = bootstrap_paired_difference(x, y)
                    comp["accuracy_mean_diff"] = float(mean_diff)
                    comp["accuracy_ci_lower"] = float(ci_lower)
                    comp["accuracy_ci_upper"] = float(ci_upper)
                    
                    # Cohen's d
                    diff = x - y
                    if diff.std() > 0:
                        cohens_d = diff.mean() / diff.std()
                        comp["cohens_d"] = float(cohens_d)
                    else:
                        comp["cohens_d"] = 0.0
                else:
                    comp["accuracy_t_statistic"] = np.nan
                    comp["accuracy_pvalue"] = np.nan
                    comp["accuracy_mean_diff"] = np.nan
                    comp["accuracy_ci_lower"] = np.nan
                    comp["accuracy_ci_upper"] = np.nan
                    comp["cohens_d"] = np.nan
            
            comparisons.append(comp)
    
    # Multiple comparison correction
    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        
        # Apply Holm correction to p-values
        if "mcnemar_pvalue" in comp_df.columns:
            valid_p = comp_df["mcnemar_pvalue"].dropna()
            if len(valid_p) > 0:
                _, p_corrected, _, _ = multipletests(valid_p, method='holm')
                comp_df.loc[valid_p.index, "mcnemar_pvalue_corrected"] = p_corrected
        
        if "accuracy_pvalue" in comp_df.columns:
            valid_p = comp_df["accuracy_pvalue"].dropna()
            if len(valid_p) > 0:
                _, p_corrected, _, _ = multipletests(valid_p, method='holm')
                comp_df.loc[valid_p.index, "accuracy_pvalue_corrected"] = p_corrected
        
        # Save CSV
        comp_df.to_csv(stats_dir / "paired_comparisons.csv", index=False)
        
        # LaTeX table - use item final field accuracy for headline comparison
        latex = "\\begin{table}[h]\n\\centering\n\\caption{Paired Model Comparisons (Item Final Field Accuracy)}\n"
        latex += "\\label{tab:paired_comparisons}\n\\footnotesize\n"
        latex += "\\begin{tabular}{llccccc}\n\\toprule\n"
        latex += "Model 1 & Model 2 & Mean Diff & 95\\% CI & t-stat & p-value & Cohen's d \\\\\n\\midrule\n"
        
        for _, row in comp_df.iterrows():
            model1_short = get_short_model_name(row['model1'])
            model2_short = get_short_model_name(row['model2'])
            latex += f"{model1_short} & {model2_short} & "
            if "accuracy_mean_diff" in row and not pd.isna(row["accuracy_mean_diff"]):
                latex += f"{row['accuracy_mean_diff']:.4f} & "
                latex += f"[{row['accuracy_ci_lower']:.4f}, {row['accuracy_ci_upper']:.4f}] & "
                latex += f"{row['accuracy_t_statistic']:.3f} & "
                pval = row.get("accuracy_pvalue_corrected", row.get("accuracy_pvalue", np.nan))
                latex += f"{pval:.4f} & "
                latex += f"{row.get('cohens_d', np.nan):.3f} \\\\\n"
            else:
                latex += "N/A & N/A & N/A & N/A & N/A \\\\\n"
        
        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        
        with open(stats_dir / "paired_comparisons.tex", "w") as f:
            f.write(latex)
        
        return comp_df
    
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Analyze PlotChain results for paper")
    parser.add_argument("--runs_dir", type=str, default="results",
                       help="Directory containing model run subdirectories")
    parser.add_argument("--output_dir", type=str, default="analysis_output",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 100)
    print("PlotChain Results Analysis")
    print("=" * 100)
    print()
    
    # Discover model runs
    print("📁 Discovering model runs...")
    model_runs = discover_model_runs(runs_dir)
    print(f"✅ Found {len(model_runs)} model runs:")
    for name in sorted(model_runs.keys()):
        print(f"   - {name}")
    print()
    
    if len(model_runs) == 0:
        raise ValueError("No valid model runs found!")
    
    # Load all data
    print("📊 Loading model data...")
    all_data = {}
    for model_name, model_dir in model_runs.items():
        try:
            all_data[model_name] = load_model_data(model_dir, model_name)
            print(f"   ✅ {model_name}")
        except Exception as e:
            print(f"   ❌ {model_name}: {e}")
            raise
    
    print()
    
    # Validate item IDs
    print("🔍 Validating item IDs...")
    common_ids = validate_item_ids(all_data)
    print()
    
    # Compute paper numbers
    print("📈 Computing paper metrics...")
    numbers = compute_paper_numbers(all_data, common_ids)
    print()
    
    # Create output directories
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "statistics").mkdir(parents=True, exist_ok=True)
    
    # Generate outputs
    print("📝 Generating tables...")
    generate_overall_leaderboard_table(all_data, numbers, output_dir)
    generate_family_matrix_table(all_data, numbers, output_dir)
    generate_hardest_families_table(all_data, output_dir)
    generate_worst_fields_table(all_data, output_dir)
    generate_error_table(all_data, common_ids, output_dir)
    generate_tolerance_table(output_dir)
    generate_parsing_rules_documentation(output_dir)
    print("   ✅ Tables generated")
    print()
    
    print("🔍 Analyzing failure modes...")
    analyze_failure_modes(all_data, common_ids, output_dir)
    print("   ✅ Failure mode analysis complete")
    print()
    
    print("📊 Generating figures...")
    generate_figures(all_data, numbers, output_dir)
    print("   ✅ Figures generated")
    print()
    
    print("🔬 Computing paired statistics...")
    paired_stats = compute_paired_statistics(all_data, common_ids, output_dir)
    print("   ✅ Statistics computed")
    
    # Store paired comparisons in paper_numbers.json
    if paired_stats is not None and not paired_stats.empty:
        # Convert DataFrame to list of dicts for JSON serialization
        # Replace NaN with None for JSON compatibility
        paired_dict = paired_stats.replace({np.nan: None}).to_dict("records")
        numbers["paired_comparisons"] = paired_dict
        print(f"   ✅ Stored {len(numbers['paired_comparisons'])} paired comparisons")
    else:
        print("   ⚠️  No paired comparisons computed")
    print()
    
    # Save paper numbers JSON
    print("💾 Saving paper numbers...")
    with open(output_dir / "paper_numbers.json", "w") as f:
        json.dump(numbers, f, indent=2, default=str)
    print("   ✅ Saved paper_numbers.json")
    print()
    
    # Print summary
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print()
    print(f"📊 Models analyzed: {len(all_data)}")
    print(f"📋 Common items: {len(common_ids)}")
    print()
    print("🏆 Leaderboard (sorted by Item Final All-Pass Rate):")
    for i, entry in enumerate(numbers["overall_leaderboard"], 1):
        item_rate = entry.get('item_final_allpass_rate', 0) * 100
        print(f"   {i}. {entry['model']}: {item_rate:.2f}% (item final all-pass)")
    print()
    print(f"📁 Output directory: {output_dir}")
    print(f"   - Tables: {output_dir / 'tables'}")
    print(f"   - Figures: {output_dir / 'figures'}")
    print(f"   - Statistics: {output_dir / 'statistics'}")
    print(f"   - Paper numbers: {output_dir / 'paper_numbers.json'}")
    print()


def generate_tolerance_table(output_dir: Path):
    """Generate tolerance table for appendix."""
    # Import tolerance function from run_plotchain_eval
    import sys
    import importlib.util
    eval_script_path = Path(__file__).parent / "run_plotchain_eval.py"
    spec = importlib.util.spec_from_file_location("run_plotchain_eval", eval_script_path)
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    
    tol_map = eval_module.tolerances_plotread()
    
    # Organize by family
    families = {}
    for (family, field), (abs_tol, rel_tol) in sorted(tol_map.items()):
        if family not in families:
            families[family] = []
        families[family].append({
            "field": field,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
        })
    
    # Generate CSV
    rows = []
    for family in sorted(families.keys()):
        for entry in sorted(families[family], key=lambda x: x["field"]):
            rows.append({
                "Family": family,
                "Field": entry["field"],
                "Absolute Tolerance": entry["abs_tol"],
                "Relative Tolerance": entry["rel_tol"],
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "tables" / "tolerance_table.csv", index=False)
    
    # Generate LaTeX table (simpler format without multirow)
    latex = "\\begin{table}[h]\n\\centering\n\\caption{Tolerance Values for PlotRead Policy}\n"
    latex += "\\label{tab:tolerances}\n\\footnotesize\n"
    latex += "\\begin{tabular}{llcc}\n\\toprule\n"
    latex += "Family & Field & Abs. Tol. & Rel. Tol. \\\\\n\\midrule\n"
    
    current_family = None
    for _, row in df.iterrows():
        family = row['Family'].replace('_', '\\_')
        field = row['Field'].replace('_', '\\_')
        abs_tol = row['Absolute Tolerance']
        rel_tol = row['Relative Tolerance']
        
        # Format tolerances
        if abs_tol == 0.0:
            abs_str = "0"
        elif abs_tol < 0.01:
            abs_str = f"{abs_tol:.3f}"
        elif abs_tol < 1.0:
            abs_str = f"{abs_tol:.2f}"
        else:
            abs_str = f"{abs_tol:.1f}"
        
        rel_str = f"{rel_tol:.2f}" if rel_tol < 1.0 else f"{rel_tol:.1f}"
        
        # Show family name for each row (can be grouped visually)
        latex += f"{family} & {field} & {abs_str} & {rel_str} \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    
    with open(output_dir / "tables" / "tolerance_table.tex", "w") as f:
        f.write(latex)
    
    print(f"   ✅ Generated tolerance table ({len(rows)} entries)")


def generate_parsing_rules_documentation(output_dir: Path):
    """Generate parsing rules documentation."""
    rules_md = """# Model Output Parsing Rules

This document specifies the exact parsing and sanitization rules used to extract structured JSON from model outputs in PlotChain evaluation.

## JSON Extraction Process

The parsing process follows these steps in order:

1. **Direct JSON Parse**: Attempt to parse the entire response text as JSON.
   - If successful and result is a dictionary, use it.
   - Otherwise, continue to step 2.

2. **Fenced Code Block Extraction**: Search for JSON within markdown code fences.
   - Pattern: ```json ... ``` or ``` ... ```
   - Extract content between fences and attempt JSON parse.

3. **First JSON Blob Extraction**: Extract first `{...}` block using regex.
   - Pattern: `\\{.*\\}` (with DOTALL flag to match across lines)
   - Attempt JSON parse on extracted blob.

## Sanitization Rules

Before parsing, the following sanitization rules are applied:

### 1. Fraction Conversion

**Pattern**: ``(\\d+(?:\\.\\d+)?)\\s*/\\s*(\\d+(?:\\.\\d+)?)``
- Matches: `a/b` where `a` and `b` are numbers (integers or decimals)
- Not preceded or followed by digits (to avoid matching dates like `2024/01/15`)
- **Replacement**: Compute `a / b` as float
  - If `b == 0`, replace with `null`
  - Otherwise, replace with computed value formatted as `{val:.12g}`

**Examples**:
- `"1025/615"` → `"1.66666666667"`
- `"3.14/2.0"` → `"1.57"`
- `"10/0"` → `"null"`

### 2. Trailing Comma Removal

**Pattern**: `,\\s*([}\\]])`
- Matches: Trailing commas before closing braces or brackets
- **Replacement**: Remove the comma

**Examples**:
- `{"a": 1,}` → `{"a": 1}`
- `[1, 2, 3,]` → `[1, 2, 3]`

### 3. Type Conversion

After JSON parsing, field values are converted using `_to_float()`:

- **Numbers** (int/float): Convert to float (preserve NaN)
- **Strings**: 
  - Strip whitespace
  - If empty → `None`
  - Otherwise, attempt `float()` conversion
    - Success → return float
    - Failure → return `None`
- **Other types**: Return `None`

## Scoring Logic

A field passes if:
- `pred` is not `None` AND `gold` is not `None` AND
- `(abs_err <= abs_tol) OR (rel_err <= rel_tol)`

Where:
- `abs_err = abs(pred - gold)`
- `rel_err = abs(pred - gold) / max(abs(gold), 1e-12)`

## Implementation Reference

These rules are implemented in:
- `run_plotchain_eval.py`: `extract_first_json()`, `_sanitize_json_candidate()`, `_to_float()`
- See source code for exact regex patterns and edge case handling.
"""
    
    with open(output_dir / "PARSING_RULES.md", "w") as f:
        f.write(rules_md)
    
    # Also generate LaTeX version for appendix
    latex = """\\section{Parsing Rules}

The following rules are applied to extract structured JSON from model outputs:

\\subsection{JSON Extraction Process}

\\begin{enumerate}
\\item \\textbf{Direct JSON Parse}: Attempt to parse the entire response as JSON.
\\item \\textbf{Fenced Code Block}: Extract JSON from markdown code fences (\\texttt{```json ... ```}).
\\item \\textbf{First JSON Blob}: Extract first \\texttt{\\{...\\}} block using regex pattern \\texttt{\\\\{.*\\}}.
\\end{enumerate}

\\subsection{Sanitization Rules}

\\subsubsection{Fraction Conversion}
Pattern: \\texttt{(\\textbackslash d+(?:\\textbackslash.\\textbackslash d+)?)\\textbackslash s*/\\textbackslash s*(\\textbackslash d+(?:\\textbackslash.\\textbackslash d+)?)}
\\begin{itemize}
\\item Matches fractions like \\texttt{"1025/615"}
\\item Replaces with computed float: \\texttt{"1.66666666667"}
\\item If denominator is zero, replaces with \\texttt{null}
\\end{itemize}

\\subsubsection{Trailing Comma Removal}
Pattern: \\texttt{,\\textbackslash s*([\\}\\]])}
\\begin{itemize}
\\item Removes trailing commas before closing braces/brackets
\\item Example: \\texttt{\\{\\"a\\": 1,\\}} → \\texttt{\\{\\"a\\": 1\\}}
\\end{itemize}

\\subsubsection{Type Conversion}
String numbers are converted to float. Empty or invalid strings become \\texttt{None}.

\\subsection{Scoring Logic}

A field passes if: \\texttt{(abs\\_err <= abs\\_tol) OR (rel\\_err <= rel\\_tol)}.
"""
    
    with open(output_dir / "tables" / "parsing_rules.tex", "w") as f:
        f.write(latex)
    
    print("   ✅ Generated parsing rules documentation")


def analyze_failure_modes(all_data: Dict[str, Dict], common_ids: List[str], output_dir: Path):
    """Analyze failure modes for bandpass_response and fft_spectrum families."""
    target_families = ["bandpass_response", "fft_spectrum"]
    
    for family in target_families:
        rows = []
        
        for model_name, data in all_data.items():
            if "per_item" not in data:
                continue
            
            per_item = data["per_item"]
            per_item_common = per_item[per_item["id"].isin(common_ids)].copy()
            family_data = per_item_common[per_item_common["type"] == family].copy()
            
            if len(family_data) == 0:
                continue
            
            # Fix NaN handling
            if "pass" in family_data.columns:
                family_data["pass"] = family_data["pass"].fillna(0).astype(float)
            
            # Group by item ID
            item_groups = family_data.groupby("id")
            
            for item_id, item_fields in item_groups:
                # Separate checkpoint and final fields
                checkpoint_fields = item_fields[item_fields["is_checkpoint"] == True] if "is_checkpoint" in item_fields.columns else pd.DataFrame()
                final_fields = item_fields[item_fields["is_checkpoint"] == False] if "is_checkpoint" in item_fields.columns else item_fields
                
                checkpoint_pass_rate = checkpoint_fields["pass"].mean() if len(checkpoint_fields) > 0 else 1.0
                final_pass_rate = final_fields["pass"].mean() if len(final_fields) > 0 else 1.0
                
                # Categorize failure mode
                if checkpoint_pass_rate < 0.5:
                    failure_mode = "Visual Interpretation Error"
                    failure_desc = "Failed to correctly read visual features from plot"
                elif final_pass_rate < 0.5:
                    failure_mode = "Numerical Reasoning Error"
                    failure_desc = "Correctly read visual features but failed in calculations"
                else:
                    failure_mode = "Partial Failure"
                    failure_desc = "Some fields failed but overall passed"
                
                # Get example failed fields
                failed_fields = final_fields[final_fields["pass"] == 0]["field"].tolist() if len(final_fields) > 0 else []
                
                rows.append({
                    "Model": model_name,
                    "Item ID": item_id,
                    "Checkpoint Pass Rate": checkpoint_pass_rate,
                    "Final Pass Rate": final_pass_rate,
                    "Failure Mode": failure_mode,
                    "Failed Fields": ", ".join(failed_fields[:3]) if failed_fields else "None",
                })
        
        if not rows:
            continue
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "tables" / f"failure_modes_{family}.csv", index=False)
        
        # Generate summary statistics
        summary_rows = []
        for model_name in df["Model"].unique():
            model_df = df[df["Model"] == model_name]
            visual_errors = (model_df["Failure Mode"] == "Visual Interpretation Error").sum()
            reasoning_errors = (model_df["Failure Mode"] == "Numerical Reasoning Error").sum()
            total_failures = len(model_df[model_df["Final Pass Rate"] < 1.0])
            
            summary_rows.append({
                "Model": model_name,
                "Total Failures": total_failures,
                "Visual Interpretation Errors": visual_errors,
                "Numerical Reasoning Errors": reasoning_errors,
                "Visual Error %": (visual_errors / total_failures * 100) if total_failures > 0 else 0,
                "Reasoning Error %": (reasoning_errors / total_failures * 100) if total_failures > 0 else 0,
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "tables" / f"failure_modes_{family}_summary.csv", index=False)
        
        # Generate LaTeX table
        latex = f"\\begin{{table}}[h]\n\\centering\n\\caption{{Failure Mode Analysis: {family.replace('_', ' ').title()}}}\n"
        latex += f"\\label{{tab:failure_modes_{family}}}\n\\footnotesize\n"
        latex += "\\begin{tabular}{lcccl}\n\\toprule\n"
        latex += "Model & Visual Errors & Reasoning Errors & Total Failures & Visual Error \\% \\\\\n\\midrule\n"
        
        for _, row in summary_df.iterrows():
            # Use get_short_model_name if available, otherwise use full name
            try:
                model_short = get_short_model_name(row['Model'])
            except:
                model_short = row['Model']
            latex += f"{model_short} & {int(row['Visual Interpretation Errors'])} & "
            latex += f"{int(row['Numerical Reasoning Errors'])} & {int(row['Total Failures'])} & "
            latex += f"{row['Visual Error %']:.1f} \\\\\n"
        
        latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        
        with open(output_dir / "tables" / f"failure_modes_{family}.tex", "w") as f:
            f.write(latex)
        
        print(f"   ✅ Generated failure mode analysis for {family}")


if __name__ == "__main__":
    main()

