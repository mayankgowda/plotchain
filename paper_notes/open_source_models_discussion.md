# Open Source Models Discussion

## Suggested Text for Discussion Section

**Location**: Discussion section, after presenting results

**Text**:

> Our evaluation focuses on proprietary multimodal LLMs accessible via standardized APIs (OpenAI GPT-4.1 and GPT-4o, Anthropic Claude Sonnet 4.5, and Google Gemini 2.5 Pro). While open-source vision-language models (e.g., LLaVA, InstructBLIP, Qwen-VL) represent an important class of models, we excluded them from this initial evaluation for several reasons: (1) the need for local deployment and GPU resources introduces variability in hardware and software configurations that complicate fair comparison, (2) varying inference frameworks (e.g., transformers, vLLM, TensorRT) and model variants require extensive hyperparameter tuning and optimization, and (3) our focus on establishing a reproducible benchmark using standardized API interfaces ensures consistent evaluation conditions across all models. However, our evaluation framework is model-agnostic and can be extended to evaluate open-source models, which we leave as future work. The PlotChain dataset and scoring code are publicly available to facilitate such evaluations.

**Alternative Shorter Version**:

> Our evaluation focuses on proprietary multimodal LLMs accessible via standardized APIs. Open-source vision-language models (e.g., LLaVA, InstructBLIP) were excluded due to the need for local deployment, varying inference frameworks, and our focus on establishing a reproducible benchmark using standardized interfaces. However, our framework is model-agnostic and can be extended to evaluate open-source models, which we leave as future work.
