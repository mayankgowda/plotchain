# PlotChain v1 (Synthetic Engineering-Plot Benchmark)

This dataset contains synthetic-but-realistic engineering plots (PNG images) paired with:
- a text question,
- a gold chain (3–5 objective reasoning steps),
- numeric ground truth values.

## Contents
- `images/` : plot images
- `plotchain_v1.jsonl` : one JSON record per item
- `plotchain_v1.csv` : flattened metadata for quick iteration

## Types included (5 each; 30 total)
- `step_response`
- `bode_magnitude`
- `fft_spectrum`
- `time_waveform`
- `iv_curve`
- `transfer_characteristic`

## Suggested evaluation prompt (template)
You are given an engineering plot image and a question.
Return:
1) Final numeric answers only (with units where applicable).
2) A short list of reasoning steps.

Be concise. Do not invent measurements not supported by the plot.

## Notes
- All plots were generated with a fixed random seed for reproducibility.
- Ground truth is stored in `ground_truth`.
