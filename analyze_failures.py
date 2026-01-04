#!/usr/bin/env python3
"""
Detailed analysis of high failure rate fields to determine if failures are due to:
1. Evaluation issues (tolerances too strict, parsing problems)
2. Genuine model errors
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# Load data
gpt_per_item = pd.read_csv('results/gpt41_plotread/per_item.csv')
claude_per_item = pd.read_csv('results/claudesonnet45_plotread/per_item.csv')
gemini_per_item = pd.read_csv('results/gemini25pro_plotread/per_item.csv')

gpt_raw = []
with open('results/gpt41_plotread/raw_openai_gpt-4.1.jsonl') as f:
    for line in f:
        if line.strip():
            gpt_raw.append(json.loads(line))
gpt_raw_dict = {r['id']: r for r in gpt_raw}

claude_raw = []
with open('results/claudesonnet45_plotread/raw_anthropic_claude-sonnet-4-5-20250929.jsonl') as f:
    for line in f:
        if line.strip():
            claude_raw.append(json.loads(line))
claude_raw_dict = {r['id']: r for r in claude_raw}

gemini_files = list(Path('results/gemini25pro_plotread').glob('raw_gemini_gemini-2.5-pro*.jsonl'))
gemini_raw = []
if gemini_files:
    with open(gemini_files[0]) as f:
        for line in f:
            if line.strip():
                gemini_raw.append(json.loads(line))
gemini_raw_dict = {r['id']: r for r in gemini_raw}

items = []
with open('data/plotchain_v4/plotchain_v4.jsonl') as f:
    for line in f:
        if line.strip():
            items.append(json.loads(line))
item_dict = {it['id']: it for it in items}

def analyze_field(family, field_name):
    print(f"\n{'=' * 100}")
    print(f"FIELD: {family}.{field_name}")
    print(f"{'=' * 100}")
    
    # Get all items for this field
    gpt_field = gpt_per_item[(gpt_per_item['type'] == family) & (gpt_per_item['field'] == field_name)]
    claude_field = claude_per_item[(claude_per_item['type'] == family) & (claude_per_item['field'] == field_name)]
    gemini_field = gemini_per_item[(gemini_per_item['type'] == family) & (gemini_per_item['field'] == field_name)]
    
    # Merge
    merged = gpt_field.merge(
        claude_field[['id', 'pred', 'gold', 'abs_err', 'rel_err', 'abs_tol', 'rel_tol', 'pass']],
        on='id',
        suffixes=('_gpt', '_claude'),
        how='inner'
    ).merge(
        gemini_field[['id', 'pred', 'gold', 'abs_err', 'rel_err', 'abs_tol', 'rel_tol', 'pass']],
        on='id',
        how='inner'
    )
    
    # Check where all 3 failed
    all_failed = merged[
        (~merged['pass_gpt']) & 
        (~merged['pass_claude']) & 
        (~merged['pass'])
    ]
    
    print(f"\nTotal items: {len(merged)}")
    print(f"All 3 failed: {len(all_failed)} ({len(all_failed)/len(merged):.1%})")
    
    if len(all_failed) == 0:
        print("No common failures to analyze")
        return
    
    # Tolerance info
    print(f"\nTolerance (plotread policy):")
    print(f"  abs_tol: {all_failed['abs_tol_gpt'].iloc[0]:.6f}")
    print(f"  rel_tol: {all_failed['rel_tol_gpt'].iloc[0]:.6f}")
    
    # Error statistics
    print(f"\nError Statistics:")
    print(f"  GPT:   mean={all_failed['abs_err_gpt'].mean():.3f}, median={all_failed['abs_err_gpt'].median():.3f}, max={all_failed['abs_err_gpt'].max():.3f}")
    print(f"  Claude: mean={all_failed['abs_err_claude'].mean():.3f}, median={all_failed['abs_err_claude'].median():.3f}, max={all_failed['abs_err_claude'].max():.3f}")
    print(f"  Gemini: mean={all_failed['abs_err'].mean():.3f}, median={all_failed['abs_err'].median():.3f}, max={all_failed['abs_err'].max():.3f}")
    
    # Check if errors exceed tolerance
    print(f"\nDetailed Error vs Tolerance Analysis (first 15 failures):")
    print(f"{'ID':<25} {'GT':<12} {'GPT':<12} {'Claude':<12} {'Gemini':<12} {'GPT Err':<10} {'Claude Err':<10} {'Gemini Err':<10} {'Pass?':<10}")
    print("-" * 120)
    
    for idx, row in all_failed.head(15).iterrows():
        gt_val = row['gold_gpt']
        gpt_pred = row['pred_gpt'] if pd.notna(row['pred_gpt']) else None
        claude_pred = row['pred_claude'] if pd.notna(row['pred_claude']) else None
        gemini_pred = row['pred'] if pd.notna(row['pred']) else None
        
        gpt_err = row['abs_err_gpt'] if pd.notna(row['abs_err_gpt']) else None
        claude_err = row['abs_err_claude'] if pd.notna(row['abs_err_claude']) else None
        gemini_err = row['abs_err'] if pd.notna(row['abs_err']) else None
        
        abs_tol = row['abs_tol_gpt']
        rel_tol = row['rel_tol_gpt']
        
        # Check if would pass with tolerance
        gpt_pass = False
        if gpt_err is not None:
            gpt_pass = (gpt_err <= abs_tol) or (row['rel_err_gpt'] <= rel_tol if pd.notna(row['rel_err_gpt']) else False)
        
        claude_pass = False
        if claude_err is not None:
            claude_pass = (claude_err <= abs_tol) or (row['rel_err_claude'] <= rel_tol if pd.notna(row['rel_err_claude']) else False)
        
        gemini_pass = False
        if gemini_err is not None:
            gemini_pass = (gemini_err <= abs_tol) or (row['rel_err'] <= rel_tol if pd.notna(row['rel_err']) else False)
        
        gpt_str = f"{gpt_pred:.3f}" if gpt_pred is not None else "null"
        claude_str = f"{claude_pred:.3f}" if claude_pred is not None else "null"
        gemini_str = f"{gemini_pred:.3f}" if gemini_pred is not None else "null"
        
        gpt_err_str = f"{gpt_err:.3f}" if gpt_err is not None else "N/A"
        claude_err_str = f"{claude_err:.3f}" if claude_err is not None else "N/A"
        gemini_err_str = f"{gemini_err:.3f}" if gemini_err is not None else "N/A"
        
        pass_str = f"{'Y' if gpt_pass else 'N'}/{'Y' if claude_pass else 'N'}/{'Y' if gemini_pass else 'N'}"
        
        print(f"{row['id']:<25} {gt_val:<12.3f} {gpt_str:<12} {claude_str:<12} {gemini_str:<12} {gpt_err_str:<10} {claude_err_str:<10} {gemini_err_str:<10} {pass_str:<10}")
    
    # Check null predictions
    gpt_nulls = all_failed['pred_gpt'].isna().sum()
    claude_nulls = all_failed['pred_claude'].isna().sum()
    gemini_nulls = all_failed['pred'].isna().sum()
    print(f"\nNull predictions:")
    print(f"  GPT: {gpt_nulls}/{len(all_failed)} ({gpt_nulls/len(all_failed):.1%})")
    print(f"  Claude: {claude_nulls}/{len(all_failed)} ({claude_nulls/len(all_failed):.1%})")
    print(f"  Gemini: {gemini_nulls}/{len(all_failed)} ({gemini_nulls/len(all_failed):.1%})")
    
    # Check raw outputs for parsing issues
    print(f"\n{'=' * 100}")
    print("RAW OUTPUT INSPECTION (checking for parsing/fraction issues)")
    print(f"{'=' * 100}")
    
    sample_ids = all_failed['id'].head(5).tolist()
    for item_id in sample_ids:
        print(f"\nItem: {item_id}")
        if item_id in item_dict:
            gt_val = item_dict[item_id]['ground_truth'].get(field_name)
            print(f"  Ground Truth: {gt_val}")
        
        if item_id in gpt_raw_dict:
            raw_text = gpt_raw_dict[item_id].get('raw_text', '')
            parsed = gpt_raw_dict[item_id].get('parsed_json', {})
            raw_val = parsed.get(field_name)
            print(f"  GPT raw parsed: {raw_val} (type: {type(raw_val).__name__})")
            # Check if raw text contains fractions
            if '/' in raw_text and field_name in raw_text.lower():
                print(f"    ⚠️  Raw text contains '/' - might be fraction")
        
        if item_id in claude_raw_dict:
            raw_text = claude_raw_dict[item_id].get('raw_text', '')
            parsed = claude_raw_dict[item_id].get('parsed_json', {})
            raw_val = parsed.get(field_name)
            print(f"  Claude raw parsed: {raw_val} (type: {type(raw_val).__name__})")
            if '/' in raw_text and field_name in raw_text.lower():
                print(f"    ⚠️  Raw text contains '/' - might be fraction")
        
        if item_id in gemini_raw_dict:
            raw_text = gemini_raw_dict[item_id].get('raw_text', '')
            parsed = gemini_raw_dict[item_id].get('parsed_json')
            if parsed:
                raw_val = parsed.get(field_name)
                print(f"  Gemini raw parsed: {raw_val} (type: {type(raw_val).__name__})")
                if '/' in raw_text and field_name in raw_text.lower():
                    print(f"    ⚠️  Raw text contains '/' - might be fraction")
            else:
                print(f"  Gemini raw parsed: None (parsing failed)")
    
    # Check if errors are genuinely large or just barely over tolerance
    print(f"\n{'=' * 100}")
    print("ERROR MAGNITUDE ANALYSIS")
    print(f"{'=' * 100}")
    
    # How many are "close" vs "way off"?
    abs_tol = all_failed['abs_tol_gpt'].iloc[0]
    rel_tol = all_failed['rel_tol_gpt'].iloc[0]
    
    gpt_close = 0
    gpt_way_off = 0
    for idx, row in all_failed.iterrows():
        if pd.notna(row['abs_err_gpt']):
            if row['abs_err_gpt'] <= abs_tol * 2:  # Within 2x tolerance
                gpt_close += 1
            else:
                gpt_way_off += 1
    
    claude_close = 0
    claude_way_off = 0
    for idx, row in all_failed.iterrows():
        if pd.notna(row['abs_err_claude']):
            if row['abs_err_claude'] <= abs_tol * 2:
                claude_close += 1
            else:
                claude_way_off += 1
    
    gemini_close = 0
    gemini_way_off = 0
    for idx, row in all_failed.iterrows():
        if pd.notna(row['abs_err']):
            if row['abs_err'] <= abs_tol * 2:
                gemini_close += 1
            else:
                gemini_way_off += 1
    
    print(f"\nError magnitude distribution:")
    print(f"  GPT:   {gpt_close} close (≤2x tol), {gpt_way_off} way off (>2x tol)")
    print(f"  Claude: {claude_close} close (≤2x tol), {claude_way_off} way off (>2x tol)")
    print(f"  Gemini: {gemini_close} close (≤2x tol), {gemini_way_off} way off (>2x tol)")
    
    # Conclusion
    print(f"\n{'=' * 100}")
    print("CONCLUSION")
    print(f"{'=' * 100}")
    
    if gpt_way_off > gpt_close and claude_way_off > claude_close and gemini_way_off > gemini_close:
        print("✅ FAILURES ARE GENUINE MODEL ERRORS")
        print("   - Most errors are way off (>2x tolerance)")
        print("   - Models are making substantial mistakes, not just barely failing")
    elif gpt_close > gpt_way_off or claude_close > claude_way_off or gemini_close > gemini_way_off:
        print("⚠️  MIXED: Some failures are close to tolerance")
        print("   - Some errors might be borderline")
        print("   - But many are still way off")
    else:
        print("✅ FAILURES ARE GENUINE MODEL ERRORS")
        print("   - Errors are substantial, not evaluation issues")

# Analyze the three fields
analyze_field('fft_spectrum', 'cp_peak_ratio')
analyze_field('bandpass_response', 'cp_f1_3db_hz')
analyze_field('bandpass_response', 'bandwidth_hz')

