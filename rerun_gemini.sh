#!/bin/bash
# Re-run Gemini 2.5 Pro with temperature=0

DATE_TAG=$(date +%Y%m%d)

echo "=========================================="
echo "Re-running Gemini 2.5 Pro with temperature=0"
echo "Date tag: $DATE_TAG"
echo "=========================================="

echo ""
echo "Running Gemini 2.5 Pro..."
python3 run_plotchain_v4_eval.py \
  --jsonl data/plotchain_v4/plotchain_v4.jsonl \
  --models gemini:gemini-2.5-pro \
  --out_dir results/gemini25pro_plotread_${DATE_TAG}_temp0_rerun \
  --mode run --policy plotread --max_output_tokens 2000 --concurrent 10 --overwrite

if [ $? -eq 0 ]; then
    echo "✅ Gemini 2.5 Pro completed successfully"
    echo ""
    echo "Next steps:"
    echo "1. Run analysis: python3 analyze_temp0_results.py"
    echo "2. Check null rate (should be <5%)"
    echo "3. Update paper with improved results"
else
    echo "❌ Gemini 2.5 Pro failed"
    exit 1
fi

