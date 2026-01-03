# PlotChain v3 (generator + multimodal eval)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r plotchain-v3/requirements.txt
```

## Generate dataset (images + JSONL + validation)
```bash
python3 plotchain-v3/generate_plotchain_v3.py \
  --out_dir data/plotchain_v3 \
  --seed 0 \
  --n_per_type 15 \
  --overwrite
```

Outputs:
- data/plotchain_v3/plotchain_v3.jsonl
- data/plotchain_v3/images/.../*.png
- data/plotchain_v3/validation_report.csv
- data/plotchain_v3/dataset_analysis.csv
- data/plotchain_v3/field_stats.csv

## Run LLM evaluation (OpenAI example)
```bash
export OPENAI_API_KEY="..."
python3 plotchain-v3/run_plotchain_eval.py \
  --jsonl data/plotchain_v3/plotchain_v3.jsonl \
  --images_root data/plotchain_v3 \
  --models openai:gpt-4.1 \
  --out_dir results_v3 \
  --overwrite
```

Prompt+score only final fields (no checkpoints):
```bash
python3 plotchain-v3/run_plotchain_eval.py ... --final_only
```
