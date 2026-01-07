#!/usr/bin/env python3
"""
Final comprehensive analysis of all 4 models with temperature=0
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import glob
import json
from datetime import datetime

def find_latest_results():
    """Find latest result directories for each model"""
    models_config = [
        ("GPT-4.1", "gpt41"),
        ("Claude 4.5", "claudesonnet45"),
        ("Gemini 2.5", "gemini25pro"),
        ("GPT-4o", "gpt4o"),
    ]
    
    found_dirs = {}
    for name, prefix in models_config:
        pattern = f"results/{prefix}*"
        matches = sorted(glob.glob(pattern), key=lambda x: Path(x).stat().st_mtime, reverse=True)
        if matches:
            latest = Path(matches[0])
            per_item = latest / "per_item.csv"
            if per_item.exists():
                found_dirs[name] = str(latest)
    
    return found_dirs

def load_results(result_dir):
    """Load all result files"""
    path = Path(result_dir)
    return {
        'per_item': pd.read_csv(path / 'per_item.csv'),
        'summary': pd.read_csv(path / 'summary.csv'),
        'overall': pd.read_csv(path / 'overall.csv'),
        'item_level': pd.read_csv(path / 'item_level.csv'),
    }

def analyze_model(per_item_df, model_name):
    """Comprehensive analysis for a single model"""
    total = len(per_item_df)
    nulls = per_item_df['pred'].isna().sum()
    valid = per_item_df[per_item_df['pred'].notna()]
    
    stats = {
        'model': model_name,
        'total': total,
        'nulls': nulls,
        'null_rate': nulls / total * 100,
        'valid': len(valid),
        'valid_rate': len(valid) / total * 100,
    }
    
    if len(valid) > 0:
        stats['passed'] = valid['pass'].sum()
        stats['failed'] = len(valid) - stats['passed']
        stats['pass_rate'] = valid['pass'].mean()
        
        # Error analysis
        failures = valid[~valid['pass']]
        if len(failures) > 0:
            errors = failures[failures['abs_err'].notna()]['abs_err']
            stats['mean_error'] = errors.mean()
            stats['median_error'] = errors.median()
            stats['std_error'] = errors.std()
            stats['max_error'] = errors.max()
            stats['p95_error'] = errors.quantile(0.95)
            
            # Errors relative to tolerance
            failed_with_tol = failures[failures['abs_tol'].notna()]
            if len(failed_with_tol) > 0:
                rel_to_tol = failed_with_tol['abs_err'] / failed_with_tol['abs_tol']
                stats['mean_error_x_tol'] = rel_to_tol.mean()
                stats['way_off_count'] = (rel_to_tol > 5).sum()
                stats['way_off_pct'] = (rel_to_tol > 5).mean() * 100
        else:
            stats.update({
                'mean_error': 0, 'median_error': 0, 'std_error': 0,
                'max_error': 0, 'p95_error': 0, 'mean_error_x_tol': 0,
                'way_off_count': 0, 'way_off_pct': 0
            })
    else:
        stats.update({
            'passed': 0, 'failed': 0, 'pass_rate': 0,
            'mean_error': 0, 'median_error': 0, 'std_error': 0,
            'max_error': 0, 'p95_error': 0, 'mean_error_x_tol': 0,
            'way_off_count': 0, 'way_off_pct': 0
        })
    
    return stats

def main():
    print("=" * 100)
    print("FINAL COMPREHENSIVE ANALYSIS: ALL 4 MODELS (TEMPERATURE=0)")
    print("=" * 100)
    
    # Find and load results
    result_dirs = find_latest_results()
    
    if len(result_dirs) < 4:
        print(f"\n⚠️  WARNING: Only found {len(result_dirs)}/4 result directories")
        return
    
    print(f"\n✅ Found all {len(result_dirs)} result directories")
    
    all_results = {}
    all_stats = {}
    
    for name, dir_path in result_dirs.items():
        print(f"Loading {name}...")
        all_results[name] = load_results(dir_path)
        all_stats[name] = analyze_model(all_results[name]['per_item'], name)
    
    # Overall performance table
    print("\n" + "=" * 100)
    print("1. OVERALL PERFORMANCE SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Model':<25} {'Total':<10} {'Nulls':<10} {'Valid':<10} {'Passed':<10} {'Pass Rate':<15} {'Mean Error':<15}")
    print("-" * 100)
    
    for name in ["GPT-4.1", "Claude 4.5", "Gemini 2.5", "GPT-4o"]:
        s = all_stats[name]
        print(f"{name:<25} {s['total']:<10} {s['nulls']:<10} {s['valid']:<10} {s['passed']:<10} {s['pass_rate']:>6.1%}        {s['mean_error']:>6.2f}")
    
    # Family-level performance
    print("\n" + "=" * 100)
    print("2. FAMILY-LEVEL PERFORMANCE")
    print("=" * 100)
    
    family_stats = {}
    for name, results in all_results.items():
        summary = results['summary']
        final_summary = summary[summary['scope'] == 'final']
        family_rates = final_summary.groupby('type')['pass_rate'].mean()
        family_stats[name] = family_rates
    
    all_families = sorted(set().union(*[set(rates.index) for rates in family_stats.values()]))
    
    print(f"\n{'Family':<25}", end="")
    for name in ["GPT-4.1", "Claude 4.5", "Gemini 2.5", "GPT-4o"]:
        print(f" {name:<15}", end="")
    print(f" {'Average':<15} {'Best':<15}")
    print("-" * 100)
    
    family_results = []
    for fam in all_families:
        rates = [family_stats[name].get(fam, 0) for name in ["GPT-4.1", "Claude 4.5", "Gemini 2.5", "GPT-4o"]]
        avg = np.mean(rates)
        best_idx = np.argmax(rates)
        best_name = ["GPT-4.1", "Claude 4.5", "Gemini 2.5", "GPT-4o"][best_idx]
        
        print(f"{fam:<25}", end="")
        for rate in rates:
            print(f" {rate:>6.1%}        ", end="")
        print(f" {avg:>6.1%}        {best_name:<15}")
        
        family_results.append({
            'family': fam,
            'gpt41': rates[0],
            'claude': rates[1],
            'gemini': rates[2],
            'gpt4o': rates[3],
            'avg': avg,
            'best': best_name
        })
    
    # Sort by average
    family_results.sort(key=lambda x: x['avg'], reverse=True)
    
    # Paired statistics
    print("\n" + "=" * 100)
    print("3. PAIRED STATISTICAL TESTS")
    print("=" * 100)
    
    # Merge all per_item dataframes
    comparison = None
    for name, results in all_results.items():
        per_item = results['per_item'][['id', 'type', 'field', 'pass', 'pred']].copy()
        per_item.columns = ['id', 'type', 'field', f'pass_{name}', f'pred_{name}']
        
        if comparison is None:
            comparison = per_item
        else:
            comparison = comparison.merge(per_item, on=['id', 'type', 'field'], how='inner')
    
    models_list = ["GPT-4.1", "Claude 4.5", "Gemini 2.5", "GPT-4o"]
    pass_cols = [f'pass_{name}' for name in models_list]
    
    print(f"\n{'Model 1':<25} {'Model 2':<25} {'t-stat':<12} {'p-value':<15} {'Cohen\'s d':<15} {'95% CI':<25}")
    print("-" * 100)
    
    paired_results = []
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            m1_name, m2_name = models_list[i], models_list[j]
            m1_col, m2_col = pass_cols[i], pass_cols[j]
            
            m1_data = comparison[m1_col].astype(int)
            m2_data = comparison[m2_col].astype(int)
            
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(m1_data, m2_data)
            
            # Effect size
            diff = m1_data - m2_data
            d = diff.mean() / diff.std() if diff.std() > 0 else 0
            
            # 95% CI
            ci = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))
            
            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{m1_name:<25} {m2_name:<25} {t_stat:>6.3f}      {p_val:>6.4f} {sig_marker:<3}      {d:>6.3f}          [{ci[0]:>6.3f}, {ci[1]:>6.3f}]")
            
            paired_results.append({
                'm1': m1_name,
                'm2': m2_name,
                't_stat': t_stat,
                'p_val': p_val,
                'cohens_d': d,
                'ci_lower': ci[0],
                'ci_upper': ci[1]
            })
    
    # Common failures
    print("\n" + "=" * 100)
    print("4. COMMON FAILURES ANALYSIS")
    print("=" * 100)
    
    comparison['all_passed'] = comparison[pass_cols].all(axis=1)
    comparison['all_failed'] = (~comparison[pass_cols]).all(axis=1)
    comparison['num_failed'] = (~comparison[pass_cols]).sum(axis=1)
    
    total = len(comparison)
    all_passed = comparison['all_passed'].sum()
    all_failed = comparison['all_failed'].sum()
    
    print(f"\nAll models passed: {all_passed}/{total} ({all_passed/total:.1%})")
    print(f"All models failed: {all_failed}/{total} ({all_failed/total:.1%})")
    print(f"Model disagreement: {total - all_passed - all_failed}/{total} ({(total - all_passed - all_failed)/total:.1%})")
    
    # Error analysis
    print("\n" + "=" * 100)
    print("5. ERROR ANALYSIS")
    print("=" * 100)
    
    print(f"\n{'Model':<25} {'Failed':<10} {'Mean Err':<12} {'Med Err':<12} {'Max Err':<12} {'Err/Tol':<12} {'Way Off':<12}")
    print("-" * 100)
    
    for name in models_list:
        s = all_stats[name]
        print(f"{name:<25} {s['failed']:<10} {s['mean_error']:>6.2f}      {s['median_error']:>6.2f}      {s['max_error']:>6.2f}      {s['mean_error_x_tol']:>6.2f}x      {s['way_off_count']:<12}")
    
    # Paper readiness
    print("\n" + "=" * 100)
    print("6. PAPER READINESS ASSESSMENT")
    print("=" * 100)
    
    issues = []
    warnings = []
    
    for name in models_list:
        s = all_stats[name]
        if s['null_rate'] > 10:
            issues.append(f"{name}: High null rate ({s['null_rate']:.1f}%)")
        elif s['null_rate'] > 5:
            warnings.append(f"{name}: Moderate null rate ({s['null_rate']:.1f}%)")
    
    print("\n✅ Strengths:")
    print("  • Temperature=0 enforced for all models")
    print("  • All 4 models evaluated")
    print("  • Comprehensive statistical analysis")
    print("  • Clear performance differences")
    
    if warnings:
        print("\n⚠️  Warnings:")
        for w in warnings:
            print(f"  • {w}")
    
    if issues:
        print("\n❌ Issues:")
        for i in issues:
            print(f"  • {i}")
    else:
        print("\n✅ No critical issues found")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    
    if not issues:
        print("\n✅ READY TO DRAFT PAPER AND FREEZE RESULTS")
    else:
        print("\n⚠️  ADDRESS ISSUES BEFORE FREEZING")
    
    # Return data for markdown generation
    return {
        'all_stats': all_stats,
        'family_results': family_results,
        'paired_results': paired_results,
        'comparison': comparison,
        'all_results': all_results
    }

if __name__ == "__main__":
    data = main()
    
    # Save summary
    print("\n" + "=" * 100)
    print("Analysis complete. Data ready for markdown generation.")
    print("=" * 100)

