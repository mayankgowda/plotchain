# Generalization Discussion

## Suggested Text for Discussion Section

**Location**: Discussion section, after presenting results and before limitations

**Text**:

> **Generalization to Real-World Plots**: While PlotChain uses synthetic plots with perfect visual quality, real-world engineering plots vary significantly in presentation quality. Scanned documents may introduce noise, compression artifacts, or resolution limitations. Hand-drawn annotations, varying fonts, non-standard axis orientations, or overlapping text could also affect model performance. However, the checkpoint-based evaluation framework provides a diagnostic tool to identify whether failures stem from visual interpretation (checkpoint failures) versus numerical reasoning (final field failures despite passing checkpoints). Models that successfully pass checkpoints on synthetic plots demonstrate correct visual feature extraction, which suggests they may generalize better to real-world images where visual quality is degraded but the underlying plot structure remains intact. Conversely, models that fail checkpoints may struggle more with real-world images due to fundamental visual interpretation limitations. Future work should evaluate this hypothesis on a dataset of real-world engineering plots with varying quality levels, compression artifacts, and presentation styles to quantify the performance gap between synthetic and real-world conditions.

**Alternative Shorter Version**:

> **Generalization to Real-World Plots**: Real-world plots vary in quality (scanned, compressed, low-resolution) compared to our synthetic dataset. However, the checkpoint-based framework helps identify whether failures stem from visual interpretation (checkpoint failures) versus numerical reasoning (final field failures). Models that pass checkpoints demonstrate correct visual feature extraction, suggesting better generalization potential. Future work should evaluate performance on real-world plot images to quantify the gap between synthetic and real-world conditions.
