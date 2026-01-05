#!/bin/bash
# Re-run all models with temperature=0 and plotread policy

DATE_TAG=$(date +%Y%m%d)

echo "=========================================="
echo "Re-running all models with temperature=0"
echo "Date tag: $DATE_TAG"
echo "=========================================="

echo ""
echo "Running GPT-4.1..."
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4.1 \
  --out_dir results/gpt41_plotread_${DATE_TAG}_temp0 \
  --mode run --policy plotread --max_output_tokens 2000 --concurrent 100 --overwrite

echo ""
echo "Running Claude Sonnet 4.5..."
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models anthropic:claude-sonnet-4-5-20250929 \
  --out_dir results/claudesonnet45_plotread_${DATE_TAG}_temp0 \
  --mode run --policy plotread --max_output_tokens 2000 --concurrent 20 --overwrite

echo ""
echo "Running Gemini 2.5 Pro..."
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.5-pro \
  --out_dir results/gemini25pro_plotread_${DATE_TAG}_temp0 \
  --mode run --policy plotread --max_output_tokens 2000 --concurrent 30 --overwrite

echo ""
echo "Running GPT-4o..."
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models openai:gpt-4o \
  --out_dir results/gpt4o_plotread_${DATE_TAG}_temp0 \
  --mode run --policy plotread --max_output_tokens 2000 --concurrent 100 --overwrite

echo ""
echo "=========================================="
echo "All runs complete!"
echo "=========================================="
