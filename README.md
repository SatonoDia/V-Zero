# V-Zero: Self-Improving Multimodal Reasoning with Zero Annotation
Recent advances in multimodal learning have significantly enhanced the reasoning capabilities of vision-language models (VLMs). However, state-of-the-art approaches rely heavily on large-scale human-annotated datasets, which are costly and time-consuming to acquire. To overcome this limitation, we introduce V-Zero, a general post-training framework that facilitates self-improvement using exclusively unlabeled images. V-Zero establishes a co-evolutionary loop by instantiating two distinct roles: a Questioner and a Solver. The Questioner learns to synthesize high-quality, challenging questions by leveraging a dual-track reasoning reward that contrasts intuitive guesses with reasoned results. The Solver is optimized using pseudo-labels derived from majority voting over its own sampled responses. Both roles are trained iteratively via Group Relative Policy Optimization (GRPO), driving a cycle of mutual enhancement. Remarkably, without a single human annotation, V-Zero achieves consistent performance gains on Qwen2.5-VL-7B-Instruct, improving visual mathematical reasoning by +1.7 and general vision-centric by +2.6, demonstrating the potential of self-improvement in multimodal systems.


## 🚀Quick Start
Under refinement...
## 📊Evaluation
We use [**VLMEvalKit**](https://github.com/open-compass/VLMEvalKit), a great open-source evaluation toolkit of VLMs for evaluation.
## Acknowledgement
V-Zero is based on [**verl**](https://github.com/volcengine/verl), a flexible, efficient and production-ready RL training library, and is inspired by [**R-Zero**](https://github.com/Chengsong-Huang/R-Zero). We gratefully acknowledge their contributions.