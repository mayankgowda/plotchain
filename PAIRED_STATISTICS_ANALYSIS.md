# Paired Statistical Analysis

## Overview

This document provides paired statistical tests (more appropriate than independent tests) since the same items are evaluated across all models.

## Methodology

- **Test**: Paired t-test (`scipy.stats.ttest_rel`)
- **Effect Size**: Cohen's d for paired samples
- **Confidence Interval**: 95% CI for mean difference
- **Interpretation**: 
  - Cohen's d: <0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), >0.8 (large)

## Results

### Table: Paired t-tests (same items across models)

| Model 1 | Model 2 | t-statistic | p-value | Cohen's d | 95% CI | Interpretation |
|---------|---------|-------------|---------|-----------|--------|----------------|
| GPT-4.1 | Claude 4.5 | 5.016 | <0.0001*** | 0.121 | [0.0272, 0.0622] | Significant, negligible effect |
| GPT-4.1 | Gemini 2.5 | -2.972 | 0.0030** | -0.072 | [-0.0386, -0.0079] | Significant, negligible effect |
| GPT-4.1 | GPT-4o | 16.311 | <0.0001*** | 0.393 | [0.1539, 0.1959] | Significant, small effect |
| Claude 4.5 | Gemini 2.5 | -7.295 | <0.0001*** | -0.176 | [-0.0863, -0.0497] | Significant, negligible effect |
| Claude 4.5 | GPT-4o | 11.415 | <0.0001*** | 0.275 | [0.1078, 0.1525] | Significant, small effect |
| Gemini 2.5 | GPT-4o | 18.503 | <0.0001*** | 0.446 | [0.1771, 0.2191] | Significant, small effect |

## Key Findings

1. **GPT-4.1 vs Gemini 2.5**: 
   - p=0.0030 (significant)
   - Effect size: -0.072 (negligible)
   - **Interpretation**: Statistically significant but practically negligible difference

2. **Gemini 2.5 vs GPT-4o**:
   - p<0.0001 (highly significant)
   - Effect size: 0.446 (small)
   - **Interpretation**: Statistically and practically significant difference

3. **GPT-4.1 vs GPT-4o**:
   - p<0.0001 (highly significant)
   - Effect size: 0.393 (small)
   - **Interpretation**: Statistically and practically significant difference

## Why Paired Tests?

- **Same Items**: All models evaluated on identical 450 items
- **More Powerful**: Paired tests have more statistical power
- **Appropriate**: Accounts for item-specific difficulty
- **Rigorous**: Standard practice for within-subjects designs

## Comparison to Independent Tests

**Independent Tests** (previous analysis):
- Assumed different items per model
- Less powerful
- Higher p-values for same effect

**Paired Tests** (current analysis):
- Accounts for same items
- More powerful
- Lower p-values for same effect
- More appropriate for this design

---

**Analysis Date**: January 2026
**Method**: Paired t-test with effect sizes and confidence intervals

