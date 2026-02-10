# Synthetic Data Limitations

## Suggested Text for Limitations Section

**Location**: Limitations subsection

**Text**:

> **Synthetic Data Limitations**: The plots in PlotChain are programmatically generated, resulting in visually perfect images with high resolution (160 DPI), clean lines, and no compression artifacts. In real-world engineering applications, plots are often scanned from documents, compressed for storage or transmission, or captured at lower resolutions, introducing noise, blur, and visual artifacts. Additionally, real-world plots may exhibit non-standard fonts, hand-drawn annotations, or varying axis orientations that are not present in our synthetic dataset. Our benchmark therefore represents a "best-case scenario" for plot reading performance, establishing an upper bound on what models can achieve under ideal visual conditions. We expect that model performance on noisy, real-world images would be lower than reported here, and future work should evaluate generalization to real-world plot images with varying quality levels, compression artifacts, and presentation styles.

**Alternative Shorter Version**:

> **Synthetic Data Limitations**: PlotChain uses programmatically generated plots with perfect visual quality (high resolution, clean lines, no artifacts). Real-world engineering plots are often scanned, compressed, or low-resolution, introducing noise and artifacts. Our benchmark represents a "best-case scenario" and establishes an upper bound on performance. Future work should evaluate generalization to real-world images with varying quality levels.
