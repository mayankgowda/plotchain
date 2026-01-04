#!/usr/bin/env python3
"""
Generate comprehensive analysis for paper
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json

# Load all results
print("Loading all model results...")

gpt41_per_item = pd.read_csv('results/gpt41_plotread/per_item.csv')
gpt41_summary = pd.read_csv('results/gpt41_plotread/summary.csv')
gpt41_overall = pd.read_csv('results/gpt41_plotread/overall.csv')

claude_per_item = pd.read_csv('results/claudesonnet45_plotread/per_item.csv')
claude_summary = pd.read_csv('results/claudesonnet45_plotread/summary.csv')
claude_overall = pd.read_csv('results/claudesonnet45_plotread/overall.csv')

gemini_per_item = pd.read_csv('results/gemini25pro_plotread/per_item.csv')
gemini_summary = pd.read_csv('results/gemini25pro_plotread/summary.csv')
gemini_overall = pd.read_csv('results/gemini25pro_plotread/overall.csv')

gpt4o_exists = Path('results/gpt4o_plotread/per_item.csv').exists()
if gpt4o_exists:
    gpt4o_per_item = pd.read_csv('results/gpt4o_plotread/per_item.csv')
    gpt4o_summary = pd.read_csv('results/gpt4o_plotread/summary.csv')
    gpt4o_overall = pd.read_csv('results/gpt4o_plotread/overall.csv')
    print("✅ GPT-4o results loaded")
else:
    print("⚠️  GPT-4o results not found")
    gpt4o_per_item = None

# Load dataset
items = []
with open('data/plotchain_v4/plotchain_v4.jsonl') as f:
    for line in f:
        if line.strip():
            items.append(json.loads(line))
item_dict = {it['id']: it for it in items}

print("=" * 100)
print("COMPREHENSIVE PAPER ANALYSIS")
print("=" * 100)

# Add difficulty
gpt41_per_item['difficulty'] = gpt41_per_item['id'].map(lambda x: item_dict.get(x, {}).get('generation', {}).get('difficulty', 'unknown'))
claude_per_item['difficulty'] = claude_per_item['id'].map(lambda x: item_dict.get(x, {}).get('generation', {}).get('difficulty', 'unknown'))
gemini_per_item['difficulty'] = gemini_per_item['id'].map(lambda x: item_dict.get(x, {}).get('generation', {}).get('difficulty', 'unknown'))
if gpt4o_exists:
    gpt4o_per_item['difficulty'] = gpt4o_per_item['id'].map(lambda x: item_dict.get(x, {}).get('generation', {}).get('difficulty', 'unknown'))

# 1. Overall Performance
print("\n" + "=" * 100)
print("1. OVERALL PERFORMANCE")
print("=" * 100)

models_data = [
    ("GPT-4.1", gpt41_per_item, gpt41_overall),
    ("Claude Sonnet 4.5", claude_per_item, claude_overall),
    ("Gemini 2.5 Pro", gemini_per_item, gemini_overall),
]
if gpt4o_exists:
    models_data.append(("GPT-4o", gpt4o_per_item, gpt4o_overall))

print("\nTable 1: Overall Performance Metrics")
print("-" * 100)
print(f"{'Model':<25} {'Overall':<12} {'Final':<12} {'CP':<12} {'Mean Error':<15} {'Null Rate':<12}")
print("-" * 100)

overall_stats = []
for name, per_item, overall in models_data:
    overall_rate = per_item['pass'].mean()
    final_rate = per_item[~per_item['field'].str.startswith('cp_')]['pass'].mean()
    cp_rate = per_item[per_item['field'].str.startswith('cp_')]['pass'].mean()
    mean_err = per_item[per_item['abs_err'].notna()]['abs_err'].mean()
    null_rate = per_item['pred'].isna().mean()
    
    print(f"{name:<25} {overall_rate:>6.1%}      {final_rate:>6.1%}      {cp_rate:>6.1%}      {mean_err:>6.2f}          {null_rate:>6.1%}")
    
    overall_stats.append({
        'model': name,
        'overall': overall_rate,
        'final': final_rate,
        'cp': cp_rate,
        'mean_error': mean_err,
        'null_rate': null_rate,
        'data': per_item['pass'].astype(int).values
    })

# Statistical significance
print("\n" + "=" * 100)
print("Statistical Significance Tests")
print("=" * 100)

print(f"\n{'Model 1':<25} {'Model 2':<25} {'t-stat':<12} {'p-value':<15} {'Significant':<15}")
print("-" * 100)

for i in range(len(overall_stats)):
    for j in range(i+1, len(overall_stats)):
        m1, m2 = overall_stats[i]['model'], overall_stats[j]['model']
        t_stat, p_val = stats.ttest_ind(overall_stats[i]['data'], overall_stats[j]['data'])
        sig = "Yes" if p_val < 0.05 else "No"
        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{m1:<25} {m2:<25} {t_stat:>6.3f}      {p_val:>6.4f} {sig_marker:<3}      {sig:<15}")

# 2. Family-Level Performance
print("\n" + "=" * 100)
print("2. FAMILY-LEVEL PERFORMANCE")
print("=" * 100)

gpt41_final = gpt41_summary[gpt41_summary['scope'] == 'final']
claude_final = claude_summary[claude_summary['scope'] == 'final']
gemini_final = gemini_summary[gemini_summary['scope'] == 'final']
if gpt4o_exists:
    gpt4o_final = gpt4o_summary[gpt4o_summary['scope'] == 'final']

gpt41_fam = gpt41_final.groupby('type')['pass_rate'].mean()
claude_fam = claude_final.groupby('type')['pass_rate'].mean()
gemini_fam = gemini_final.groupby('type')['pass_rate'].mean()
if gpt4o_exists:
    gpt4o_fam = gpt4o_final.groupby('type')['pass_rate'].mean()

all_families = sorted(set(gpt41_fam.index) | set(claude_fam.index) | set(gemini_fam.index))
if gpt4o_exists:
    all_families = sorted(set(all_families) | set(gpt4o_fam.index))

print("\nTable 2: Family-Level Pass Rates (Final Fields)")
print("-" * 100)
header = f"{'Family':<25} {'GPT-4.1':<12} {'Claude 4.5':<12} {'Gemini 2.5':<12}"
if gpt4o_exists:
    header += f" {'GPT-4o':<12}"
header += f" {'Average':<12} {'Std Dev':<12} {'Best':<15}"
print(header)
print("-" * 100)

family_results = []
for fam in all_families:
    rates = []
    gpt41_rate = gpt41_fam.get(fam, 0)
    claude_rate = claude_fam.get(fam, 0)
    gemini_rate = gemini_fam.get(fam, 0)
    rates.extend([gpt41_rate, claude_rate, gemini_rate])
    
    if gpt4o_exists:
        gpt4o_rate = gpt4o_fam.get(fam, 0)
        rates.append(gpt4o_rate)
    
    avg = np.mean(rates)
    std = np.std(rates)
    best_idx = np.argmax(rates)
    best_names = ["GPT-4.1", "Claude 4.5", "Gemini 2.5"]
    if gpt4o_exists:
        best_names.append("GPT-4o")
    best_name = best_names[best_idx]
    
    row = f"{fam:<25} {gpt41_rate:>6.1%}      {claude_rate:>6.1%}      {gemini_rate:>6.1%}"
    if gpt4o_exists:
        row += f"      {gpt4o_rate:>6.1%}"
    row += f"      {avg:>6.1%}      {std:>6.3f}      {best_name:<15}"
    print(row)
    
    family_results.append({
        'family': fam,
        'gpt41': gpt41_rate,
        'claude': claude_rate,
        'gemini': gemini_rate,
        'gpt4o': gpt4o_rate if gpt4o_exists else None,
        'avg': avg,
        'std': std,
        'best': best_name
    })

# Sort by average
family_results.sort(key=lambda x: x['avg'], reverse=True)
print("\nRanked by Average Pass Rate:")
for i, fr in enumerate(family_results, 1):
    print(f"{i:2d}. {fr['family']:<25} {fr['avg']:>6.1%} (best: {fr['best']})")

# 3. Common Failures
print("\n" + "=" * 100)
print("3. COMMON FAILURES ANALYSIS")
print("=" * 100)

comparison = gpt41_per_item.merge(
    claude_per_item[['id', 'type', 'field', 'pass', 'pred']],
    on=['id', 'type', 'field'],
    suffixes=('_gpt41', '_claude'),
    how='inner'
).merge(
    gemini_per_item[['id', 'type', 'field', 'pass', 'pred']],
    on=['id', 'type', 'field'],
    suffixes=('', '_gemini'),
    how='inner'
)

if gpt4o_exists:
    comparison = comparison.merge(
        gpt4o_per_item[['id', 'type', 'field', 'pass', 'pred']],
        on=['id', 'type', 'field'],
        suffixes=('', '_gpt4o'),
        how='inner'
    )

comparison['gpt41_failed'] = (~comparison['pass_gpt41']) | comparison['pred_gpt41'].isna()
comparison['claude_failed'] = (~comparison['pass_claude']) | comparison['pred_claude'].isna()
comparison['gemini_failed'] = (~comparison['pass']) | comparison['pred'].isna()

if gpt4o_exists:
    comparison['gpt4o_failed'] = (~comparison['pass_gpt4o']) | comparison['pred_gpt4o'].isna()
    comparison['all_failed'] = comparison['gpt41_failed'] & comparison['claude_failed'] & comparison['gemini_failed'] & comparison['gpt4o_failed']
    comparison['all_passed'] = (~comparison['gpt41_failed']) & (~comparison['claude_failed']) & (~comparison['gemini_failed']) & (~comparison['gpt4o_failed'])
    comparison['num_failed'] = comparison['gpt41_failed'].astype(int) + comparison['claude_failed'].astype(int) + comparison['gemini_failed'].astype(int) + comparison['gpt4o_failed'].astype(int)
else:
    comparison['all_failed'] = comparison['gpt41_failed'] & comparison['claude_failed'] & comparison['gemini_failed']
    comparison['all_passed'] = (~comparison['gpt41_failed']) & (~comparison['claude_failed']) & (~comparison['gemini_failed'])
    comparison['num_failed'] = comparison['gpt41_failed'].astype(int) + comparison['claude_failed'].astype(int) + comparison['gemini_failed'].astype(int)

total = len(comparison)
all_passed_count = comparison['all_passed'].sum()
all_failed_count = comparison['all_failed'].sum()

print("\nTable 3: Common Failure Patterns")
print("-" * 100)
print(f"{'Pattern':<40} {'Count':<15} {'Percentage':<15}")
print("-" * 100)
print(f"{'All models passed':<40} {all_passed_count:<15} {all_passed_count/total:>6.1%}")
print(f"{'All models failed':<40} {all_failed_count:<15} {all_failed_count/total:>6.1%}")

if gpt4o_exists:
    print(f"{'Exactly 3 models failed':<40} {(comparison['num_failed'] == 3).sum():<15} {(comparison['num_failed'] == 3).sum()/total:>6.1%}")
print(f"{'Exactly 2 models failed':<40} {(comparison['num_failed'] == 2).sum():<15} {(comparison['num_failed'] == 2).sum()/total:>6.1%}")
print(f"{'Exactly 1 model failed':<40} {(comparison['num_failed'] == 1).sum():<15} {(comparison['num_failed'] == 1).sum()/total:>6.1%}")

# By family
print("\nTable 4: Common Failures by Family")
print("-" * 100)
print(f"{'Family':<25} {'All Pass':<15} {'All Fail':<15} {'Consensus':<15}")
print("-" * 100)

for fam in sorted(comparison['type'].unique()):
    fam_data = comparison[comparison['type'] == fam]
    fam_total = len(fam_data)
    fam_all_pass = fam_data['all_passed'].sum()
    fam_all_fail = fam_data['all_failed'].sum()
    consensus = (fam_all_pass + fam_all_fail) / fam_total if fam_total > 0 else 0
    
    print(f"{fam:<25} {fam_all_pass:>4}/{fam_total:<4} ({fam_all_pass/fam_total:>5.1%})  {fam_all_fail:>4}/{fam_total:<4} ({fam_all_fail/fam_total:>5.1%})  {consensus:>6.1%}")

# 4. Difficulty Impact
print("\n" + "=" * 100)
print("4. DIFFICULTY IMPACT ANALYSIS")
print("=" * 100)

print("\nTable 5: Performance by Difficulty Level")
print("-" * 100)
header = f"{'Difficulty':<15} {'GPT-4.1':<12} {'Claude 4.5':<12} {'Gemini 2.5':<12}"
if gpt4o_exists:
    header += f" {'GPT-4o':<12}"
header += f" {'Average':<12}"
print(header)
print("-" * 100)

for diff in ['clean', 'moderate', 'edge']:
    gpt41_diff = gpt41_per_item[gpt41_per_item['difficulty'] == diff]['pass'].mean()
    claude_diff = claude_per_item[claude_per_item['difficulty'] == diff]['pass'].mean()
    gemini_diff = gemini_per_item[gemini_per_item['difficulty'] == diff]['pass'].mean()
    
    rates = [gpt41_diff, claude_diff, gemini_diff]
    if gpt4o_exists:
        gpt4o_diff = gpt4o_per_item[gpt4o_per_item['difficulty'] == diff]['pass'].mean()
        rates.append(gpt4o_diff)
        print(f"{diff:<15} {gpt41_diff:>6.1%}      {claude_diff:>6.1%}      {gemini_diff:>6.1%}      {gpt4o_diff:>6.1%}      {np.mean(rates):>6.1%}")
    else:
        print(f"{diff:<15} {gpt41_diff:>6.1%}      {claude_diff:>6.1%}      {gemini_diff:>6.1%}      {np.mean(rates):>6.1%}")

# ANOVA
print("\nStatistical Test: Difficulty Impact (ANOVA)")
print("-" * 100)
print(f"{'Model':<25} {'F-statistic':<15} {'p-value':<15} {'Significant':<15}")
print("-" * 100)

for name, per_item in [("GPT-4.1", gpt41_per_item), ("Claude 4.5", claude_per_item), ("Gemini 2.5", gemini_per_item)]:
    clean = per_item[per_item['difficulty'] == 'clean']['pass'].astype(int)
    moderate = per_item[per_item['difficulty'] == 'moderate']['pass'].astype(int)
    edge = per_item[per_item['difficulty'] == 'edge']['pass'].astype(int)
    
    f_stat, p_val = stats.f_oneway(clean, moderate, edge)
    sig = "Yes" if p_val < 0.05 else "No"
    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{name:<25} {f_stat:>6.3f}          {p_val:>6.4f} {sig_marker:<3}      {sig:<15}")

if gpt4o_exists:
    clean = gpt4o_per_item[gpt4o_per_item['difficulty'] == 'clean']['pass'].astype(int)
    moderate = gpt4o_per_item[gpt4o_per_item['difficulty'] == 'moderate']['pass'].astype(int)
    edge = gpt4o_per_item[gpt4o_per_item['difficulty'] == 'edge']['pass'].astype(int)
    f_stat, p_val = stats.f_oneway(clean, moderate, edge)
    sig = "Yes" if p_val < 0.05 else "No"
    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{'GPT-4o':<25} {f_stat:>6.3f}          {p_val:>6.4f} {sig_marker:<3}      {sig:<15}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

