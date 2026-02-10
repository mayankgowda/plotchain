# Reviewer Response Plan

This document outlines the changes needed to address reviewer comments for the PlotChain paper.

## Review 1 Comments

### 1. Clarify the Tolerances
**Request**: Add a table in the Appendix listing tolerances for each plot family (e.g., "For Bode plots, we allow ±0.5 dB").

**Action Required**:
- Create a LaTeX table in the appendix documenting all tolerance values
- Include both absolute and relative tolerances for each (family, field) pair
- Format: Family | Field | Absolute Tolerance | Relative Tolerance | Units/Notes
- Use the `plotread` policy tolerances (as that's what was used in the paper)

**Files to Create/Modify**:
- Create: `analysis_output/tables/tolerance_table.tex` (LaTeX table)
- Create: `analysis_output/tables/tolerance_table.csv` (CSV version)
- Modify: `analyze_plotchain_results.py` - Add function to generate tolerance table

**Implementation Notes**:
- Extract tolerances from `tolerances_plotread()` function in `run_plotchain_eval.py`
- Group by family for readability
- Include units in field names or add a units column
- Note that tolerances are applied as: `pass = (abs_err <= abs_tol) OR (rel_err <= rel_tol)`

---

### 2. Open Source Models
**Request**: Add a sentence or two in the "Discussion" explaining why open-source models (like LLaVA) were excluded, or perhaps include one if it is easy to run.

**Action Required**:
- Add a paragraph in the Discussion/Limitations section explaining:
  - Focus on proprietary models for initial benchmark establishment
  - API availability and standardization considerations
  - Note that open-source models can be evaluated using the same codebase
  - Future work could include open-source models

**Files to Create/Modify**:
- This is a paper text change (not code), but we can create a note file:
- Create: `paper_notes/open_source_models_discussion.md` (draft text for paper)

**Suggested Text**:
> "Our evaluation focuses on proprietary multimodal LLMs accessible via standardized APIs (OpenAI, Anthropic, Google). While open-source vision-language models (e.g., LLaVA, InstructBLIP) represent an important class of models, we excluded them from this initial evaluation due to: (1) the need for local deployment and GPU resources, (2) varying inference frameworks and configurations that complicate fair comparison, and (3) our focus on establishing a reproducible benchmark using standardized API interfaces. However, our evaluation framework is model-agnostic and can be extended to evaluate open-source models, which we leave as future work."

---

### 3. Synthetic vs. Real Data - Limitations Section
**Request**: Acknowledge in the "Limitations" section that this benchmark represents a "best-case scenario" and that performance on noisy, real-world images might be lower.

**Action Required**:
- Add a Limitations subsection discussing:
  - Synthetic plots are visually perfect (high resolution, clean lines, no artifacts)
  - Real-world plots may be scanned, compressed, low-resolution, or contain noise
  - Performance on real-world images may be lower
  - This benchmark establishes an upper bound on performance

**Files to Create/Modify**:
- Create: `paper_notes/limitations_synthetic_data.md` (draft text for paper)

**Suggested Text**:
> "**Synthetic Data Limitations**: The plots in PlotChain are programmatically generated, resulting in visually perfect images with high resolution, clean lines, and no compression artifacts. In real-world engineering applications, plots are often scanned from documents, compressed for storage, or captured at lower resolutions, introducing noise and visual artifacts. Our benchmark therefore represents a "best-case scenario" for plot reading performance. We expect that model performance on noisy, real-world images would be lower than reported here, and future work should evaluate generalization to real-world plot images with varying quality levels."

---

### 4. Parsing Rules
**Request**: Explicitly state what rules you used for sanitizing model outputs (e.g., fixing fractions) so that other researchers can replicate your scoring method exactly.

**Action Required**:
- Document all parsing/sanitization rules used in `extract_first_json()` and `_sanitize_json_candidate()`
- Create a clear specification document
- Add to appendix or supplementary material

**Files to Create/Modify**:
- Create: `PARSING_RULES.md` (detailed specification)
- Create: `analysis_output/tables/parsing_rules.tex` (LaTeX table for paper appendix)

**Rules to Document**:
1. **JSON Extraction**:
   - Try direct JSON parse first
   - If that fails, look for fenced code blocks: ```json ... ```
   - If that fails, extract first `{...}` blob using regex
   
2. **Fraction Sanitization**:
   - Pattern: `(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)` (not preceded/followed by digits)
   - Replace `a/b` with computed float: `a / b` (if `b != 0`, else `null`)
   - Example: `"1025/615"` → `"1.66666666667"`
   
3. **Trailing Comma Removal**:
   - Pattern: `,\s*([}\]])`
   - Replace trailing commas before closing braces/brackets
   - Example: `{"a": 1,}` → `{"a": 1}`
   
4. **Type Conversion**:
   - String numbers are converted to float via `_to_float()`
   - Empty strings → `None`
   - Invalid strings → `None`

---

## Review 2 Comments

### 1. Expand Failure Mode Analysis
**Request**: Expand failure mode analysis, particularly for challenging cases such as bandpass responses and FFT spectrum tasks. Additional qualitative examples or discussion of whether errors stem from visual interpretation versus numerical reasoning would strengthen the analysis.

**Action Required**:
- Analyze failure patterns for `bandpass_response` and `fft_spectrum` families
- Categorize errors as:
  - Visual interpretation errors (misreading axes, misidentifying peaks)
  - Numerical reasoning errors (incorrect calculations from correctly read values)
  - Format/compliance errors (missing fields, wrong JSON structure)
- Create qualitative examples showing common failure modes
- Add analysis function to `analyze_plotchain_results.py`

**Files to Create/Modify**:
- Modify: `analyze_plotchain_results.py` - Add `analyze_failure_modes()` function
- Create: `analysis_output/tables/failure_modes_bandpass.tex`
- Create: `analysis_output/tables/failure_modes_fft.tex`
- Create: `analysis_output/figures/failure_examples.pdf` (if visual examples are needed)

**Implementation Approach**:
- For each failing item in `bandpass_response` and `fft_spectrum`:
  - Compare predicted vs. ground truth values
  - Check if checkpoint fields passed (visual reading OK) but final fields failed (reasoning error)
  - Categorize error type
  - Extract representative examples

---

### 2. Tolerance Policy Justification
**Request**: Brief justification of the tolerance policy used for scoring (e.g., whether thresholds are based on human studies, engineering standards, or expert judgment).

**Action Required**:
- Add a paragraph in the Methodology section explaining:
  - Tolerance policy rationale ("plotread" = fair human plot-read tolerances)
  - Based on expert judgment of reasonable reading precision
  - Designed to allow for typical human measurement uncertainty
  - Not based on formal human studies (acknowledge this limitation)

**Files to Create/Modify**:
- Create: `paper_notes/tolerance_justification.md` (draft text for paper)

**Suggested Text**:
> "**Tolerance Policy**: We use a "plotread" tolerance policy designed to reflect fair human plot-reading precision. Tolerances are specified as (absolute_tolerance, relative_tolerance) pairs for each (plot_family, field) combination, where a prediction passes if either the absolute error ≤ absolute_tolerance OR the relative error ≤ relative_tolerance. These thresholds were set through expert judgment to allow for typical human measurement uncertainty when reading values from axes and identifying features (e.g., ±0.5 dB for Bode plot gain, ±5 Hz for frequency measurements). While not derived from formal human studies, these tolerances are designed to distinguish between reasonable reading precision and systematic errors. A "strict" policy with tighter tolerances is also available for more stringent evaluation (see Appendix)."

---

### 3. Generalization Discussion
**Request**: Short discussion on how results may generalize to real-world engineering plots.

**Action Required**:
- Add a paragraph in Discussion section about generalization
- Connect to the limitations about synthetic data
- Discuss potential factors affecting generalization:
  - Image quality and resolution
  - Compression artifacts
  - Axis label clarity
  - Plot style variations

**Files to Create/Modify**:
- Create: `paper_notes/generalization_discussion.md` (draft text for paper)

**Suggested Text**:
> "**Generalization to Real-World Plots**: While PlotChain uses synthetic plots with perfect visual quality, real-world engineering plots vary significantly in presentation quality. Scanned documents may introduce noise, compression artifacts, or resolution limitations. Hand-drawn annotations, varying fonts, or non-standard axis orientations could also affect model performance. However, the checkpoint-based evaluation framework provides a diagnostic tool to identify whether failures stem from visual interpretation (checkpoint failures) versus numerical reasoning (final field failures despite passing checkpoints). This suggests that models that successfully pass checkpoints on synthetic plots may generalize better to real-world images, as they demonstrate correct visual feature extraction. Future work should evaluate this hypothesis on a dataset of real-world engineering plots."

---

### 4. Editorial Polish - MLLM Capitalization
**Request**: Consistent capitalization of "MLLM" (Multimodal Large Language Model).

**Action Required**:
- Standardize to "MLLM" (all caps) throughout the paper
- Check for variations: "Mllm", "mllm", "multimodal LLM", etc.
- This is a paper text change (not code)

**Files to Create/Modify**:
- This is a paper text change - create a checklist:
- Create: `paper_notes/editorial_checklist.md`

---

## Implementation Priority

### High Priority (Required for Revision)
1. ✅ Tolerance table (Appendix) - **Code + Output**
2. ✅ Parsing rules documentation - **Code + Output**
3. ✅ Failure mode analysis for bandpass/FFT - **Code + Output**
4. ✅ Tolerance justification text - **Paper text**

### Medium Priority (Strengthens Paper)
5. ✅ Synthetic data limitations text - **Paper text**
6. ✅ Generalization discussion - **Paper text**
7. ✅ Open-source models discussion - **Paper text**

### Low Priority (Editorial)
8. ✅ MLLM capitalization consistency - **Paper text**

---

## Next Steps

1. **Generate Tolerance Table**: Create script to extract and format tolerances
2. **Document Parsing Rules**: Create detailed specification document
3. **Implement Failure Mode Analysis**: Add analysis function for bandpass/FFT
4. **Draft Paper Text Sections**: Create markdown files with suggested text for each section
5. **Review and Integrate**: Review all outputs and integrate into paper draft
