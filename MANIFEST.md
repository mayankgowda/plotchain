# PlotChain Dataset Manifest

## Dataset Information

- **Dataset Name**: PlotChain
- **Version**: 1.0 (FROZEN)
- **Generation Date**: January 2026
- **Total Items**: 450
- **Families**: 15
- **Items per Family**: 30

## Artifacts

### Code Artifacts

1. **Generator Script**: `generate_plotchain.py`
   - **Commit Hash**: e3f868d9b2e30a98434aad52a9121c458aa8296e
   - **Purpose**: Generates deterministic plots and ground truth
   - **Master Seed**: 0 (hardcoded in script)
   - **SHA256**: 46a04671b5729ea8caa9ed2867ee508f90b82bd9ccc1c81a2972b71bdf55df7d

2. **Evaluation Script**: `run_plotchain_eval.py`
   - **Commit Hash**: e3f868d9b2e30a98434aad52a9121c458aa8296e
   - **Purpose**: Runs models and scores outputs
   - **Dependencies**: `concurrent_eval.py` for concurrent processing
   - **SHA256**: 3527efd0576f7ea44a7d35cb0adcd14f36ae9bba256f5e9d37c5004153368cde
   - **Temperature**: 0 (enforced in code)

3. **Concurrent Evaluation**: `concurrent_eval.py`
   - **Commit Hash**: e3f868d9b2e30a98434aad52a9121c458aa8296e
   - **Purpose**: Handles concurrent API calls with rate limiting
   - **SHA256**: 41793b676ef0edf149a23ac4f211c17d7b3230c2f901417534dc73e6886aafa3

### Data Artifacts

1. **Dataset JSONL**: `data/plotchain/plotchain.jsonl`
   - **Format**: One JSON object per line
   - **Fields**: id, type, image_path, question, ground_truth, plot_params, generation

2. **Images**: `data/plotchain/images/<family>/*.png`
   - **Format**: PNG, 160 DPI
   - **Size**: 6.0 × 3.6 inches (or 5.2 × 5.2 for pole_zero)

3. **Validation Files**: 
   - `data/plotchain/validation_rows.csv`
   - `data/plotchain/validation_summary.csv`

## Generation Parameters

- **Master Seed**: 0
- **Items per Family**: 30
- **Difficulty Split**: 40% clean, 30% moderate, 30% edge
- **Reproducibility**: 100% reproducible from master seed

## Model Evaluation Settings

### Common Settings (All Models)

- **Temperature**: 0 (deterministic)
- **Max Output Tokens**: 2000 (or default 400)
- **Policy**: plotread (fair human plot-read tolerances)
- **Prompt**: Uniform across all models (see `build_prompt()` in `run_plotchain_eval.py`)

### Model-Specific Settings

#### GPT-4.1
- **Provider**: OpenAI
- **Model**: `gpt-4.1`
- **API**: OpenAI Responses API
- **Temperature**: 0 (enforced)
- **Max Output Tokens**: 2000
- **Run Date**: 2026-01-04 10:15:54

#### Claude Sonnet 4.5
- **Provider**: Anthropic
- **Model**: `claude-sonnet-4-5-20250929`
- **API**: Anthropic Messages API
- **Temperature**: 0 (enforced)
- **Max Output Tokens**: 2000
- **Run Date**: 2026-01-04 09:37:08

#### Gemini 2.5 Pro
- **Provider**: Google
- **Model**: `gemini-2.5-pro`
- **API**: Google Generative AI API
- **Temperature**: 0 (enforced)
- **Max Output Tokens**: 2000
- **Run Date**: 2026-01-04 09:38:40

#### GPT-4o
- **Provider**: OpenAI
- **Model**: `gpt-4o`
- **API**: OpenAI Responses API
- **Temperature**: 0 (enforced)
- **Max Output Tokens**: 2000
- **Run Date**: 2026-01-04 11:47:52

## Reproducibility

### To Reproduce Dataset

```bash
python3 generate_plotchain.py \
  --out_dir data/plotchain \
  --n_per_family 30 \
  --seed 0
```

### To Reproduce Evaluation

```bash
python3 run_plotchain_eval.py \
  --jsonl data/plotchain/plotchain.jsonl \
  --models openai:gpt-4.1 anthropic:claude-sonnet-4-5-20250929 gemini:gemini-2.5-pro openai:gpt-4o \
  --out_dir results \
  --mode run \
  --policy plotread \
  --max_output_tokens 2000 \
  --concurrent 10
```

## Checksums

- **Dataset JSONL**: d5875f38c5e0f2dc2561d5973283c5c4fcc5df0e1774682d557443e6d273d141
- **Generator Script**: 46a04671b5729ea8caa9ed2867ee508f90b82bd9ccc1c81a2972b71bdf55df7d
- **Evaluation Script**: 3527efd0576f7ea44a7d35cb0adcd14f36ae9bba256f5e9d37c5004153368cde
- **Concurrent Evaluation**: 41793b676ef0edf149a23ac4f211c17d7b3230c2f901417534dc73e6886aafa3

## Notes

- All plots are generated deterministically from master seed
- Ground truth is computed from plot parameters (not OCR)
- Evaluation uses uniform tolerances across all models
- All model calls use temperature=0 for reproducibility

