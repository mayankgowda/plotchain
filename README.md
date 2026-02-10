# PlotChain: A Benchmark for Multimodal LLM Plot Reading

PlotChain is a synthetic engineering-plot benchmark for evaluating multimodal large language models (MLLMs) on their ability to read and extract quantitative information from engineering plots.

## Overview

PlotChain contains **450 synthetic-but-realistic engineering plots** (PNG images) across **15 plot families**, with each plot paired with:
- A text question asking for specific quantitative measurements
- Numeric ground truth values (computed deterministically from plot parameters)
- Checkpoint fields for diagnostic evaluation (visual reading vs. numerical reasoning)

## Dataset Statistics

- **Total Items**: 450
- **Plot Families**: 15
- **Items per Family**: 30
- **Difficulty Split**: 40% clean, 30% moderate, 30% edge cases
- **Format**: PNG images (160 DPI, 6.0 × 3.6 inches)

## Plot Families

1. **step_response** - Controls; time vs response
2. **bode_magnitude** - Circuits/controls; log-frequency vs dB
3. **bode_phase** - Circuits/controls; log-frequency vs degrees
4. **bandpass_response** - Filters; log-frequency vs dB
5. **time_waveform** - Signals; time vs voltage
6. **fft_spectrum** - Signals; frequency vs magnitude
7. **spectrogram** - Signals; time-frequency heatmap
8. **iv_resistor** - Circuits; V-I linear relationship
9. **iv_diode** - Circuits; V-I exponential relationship
10. **transfer_characteristic** - Nonlinear blocks; Vin-Vout
11. **pole_zero** - Systems; complex plane
12. **stress_strain** - Materials; strain vs stress
13. **torque_speed** - Motors; speed vs torque
14. **pump_curve** - Fluids; flow vs head
15. **sn_curve** - Fatigue; log-log cycles vs stress

See `analysis_output/tables/tolerance_table.tex` for a detailed table of families and representative outputs.

## Installation

### Requirements

- Python 3.9+
- Required packages (see `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

- For analysis scripts (see `requirements_analysis.txt`):
  ```bash
  pip install -r requirements_analysis.txt
  ```

### API Keys (for running evaluations)

Set up your API keys as environment variables:
```bash
export ANTHROPIC_API_KEY=your_key_here      # For Claude
export OPENAI_API_KEY=your_key_here         # For GPT models
export GOOGLE_API_KEY=your_key_here         # For Gemini
```

See `.env.example` for a template.

## Quick Start

### 1. Generate Dataset

```bash
python3 generate_plotchain.py \
  --out_dir data/plotchain \
  --n_per_family 30 \
  --seed 0
```

This generates:
- `data/plotchain/plotchain.jsonl` - Dataset file (one JSON object per line)
- `data/plotchain/images/<family>/*.png` - Plot images

### 2. Run Evaluation

Evaluate a model on the dataset:

```bash
python3 run_plotchain_eval.py \
  --jsonl data/plotchain/plotchain.jsonl \
  --models openai:gpt-4.1 \
  --out_dir results \
  --mode run \
  --policy plotread \
  --max_output_tokens 2000 \
  --concurrent 10 \
  --images data/plotchain
```

**Supported Models**:
- OpenAI: `gpt-4.1`, `gpt-4o`
- Anthropic: `claude-sonnet-4-5-20250929`
- Google: `gemini-2.5-pro`

**Evaluation Settings**:
- `--policy plotread`: Fair human plot-read tolerances (default)
- `--policy strict`: Stricter tolerances (~60% of plotread values)
- `--temperature`: Always 0 (deterministic, enforced in code)

### 3. Analyze Results

Generate paper-ready analysis outputs:

```bash
python3 analyze_plotchain_results.py \
  --runs_dir results \
  --output_dir analysis_output
```

This generates:
- **Tables**: LaTeX tables (leaderboard, family performance, error analysis, etc.)
- **Figures**: PDF figures (heatmaps, pass rates, accuracy vs latency)
- **Statistics**: Paired comparison statistics (McNemar tests, bootstrap CIs)
- **Paper Numbers**: `paper_numbers.json` with all key metrics

### 4. Generate Tolerance Table

Generate the families description table:

```bash
python3 generate_tolerance_table.py --output_dir analysis_output
```

Outputs: `analysis_output/tables/tolerance_table.tex`

## Dataset Structure

### JSONL Format

Each line in `plotchain.jsonl` is a JSON object:

```json
{
  "id": "step_response_001",
  "type": "step_response",
  "image_path": "images/step_response/step_response_001.png",
  "question": "From the step response plot, read...",
  "ground_truth": {
    "percent_overshoot": 15.2,
    "settling_time_s": 2.5,
    "steady_state": 1.0,
    "cp_peak_time_s": 0.8,
    "cp_peak_value": 1.15
  },
  "plot_params": {...},
  "generation": {
    "seed": 42,
    "difficulty": "clean",
    "final_fields": ["percent_overshoot", "settling_time_s", "steady_state"],
    "checkpoint_fields": ["cp_peak_time_s", "cp_peak_value"]
  }
}
```

### Field Types

- **Final Fields**: Primary outputs requested by the question (e.g., `cutoff_hz`, `resonance_hz`)
- **Checkpoint Fields** (prefixed `cp_`): Diagnostic fields for visual reading verification (e.g., `cp_peak_time_s`, `cp_mag_at_fc_db`)

## Evaluation Metrics

### Field-Level Metrics
- **Final Field Pass Rate**: Percentage of final fields that pass tolerance checks
- **Checkpoint Field Pass Rate**: Percentage of checkpoint fields that pass

### Item-Level Metrics
- **Item Final All-Pass Rate**: Percentage of items where ALL final fields pass (headline metric)
- **Item Final Field Accuracy**: Mean pass rate over final fields per item

### Scoring

A field passes if:
```
(abs_error <= abs_tolerance) OR (rel_error <= rel_tolerance)
```

Tolerances are defined per (family, field) pair. See `run_plotchain_eval.py` for tolerance definitions.

## Reproducibility

- **Deterministic Generation**: All plots generated from master seed (seed=0)
- **Deterministic Evaluation**: Temperature=0 enforced for all model calls
- **Version Control**: See `MANIFEST.md` for checksums and reproducibility details

## Citation

If you use PlotChain in your research, please cite:

```bibtex
@article{plotchain2026,
  title={PlotChain: A Benchmark for Multimodal LLM Plot Reading},
  author={...},
  journal={...},
  year={2026}
}
```

## Documentation

- **MANIFEST.md**: Complete reproducibility manifest with checksums
- **PUBLIC_RELEASE_CHECKLIST.md**: Checklist for public release preparation
- **paper_notes/**: Draft text for paper sections (tolerance justification, limitations, etc.)
- **analysis_output/PARSING_RULES.md**: Detailed parsing and sanitization rules

## License

[Add your license here]

## Contact

[Add contact information]
