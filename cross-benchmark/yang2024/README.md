# Yang, Tsai and Yamada (2024)

**Source:** Daniel Yang, Yao-Hung Hubert Tsai, Makoto Yamada. *On Verbalized Confidence Scores for LLMs.* arXiv:2412.14737. (The earlier version of this README cited arXiv:2404.09272, which is a different paper.)

## Withdrawn

The earlier `yang2024_analysis.py` contained a hard-coded results table (GPT-4, LLaMA-2, Vicuna, Mistral, Qwen-14B, …; ρ(L, AUROC) = +.894). Those models are not in Yang et al.'s released data, and the numbers were not computed from it. The script has been removed, and the results reported from it in arXiv:2604.17714v1 are withdrawn.

## Corrected analysis

`analysis/v2/yang_rescore.py` reads Yang et al.'s released item-level predictions: `predictions/y_pred/<benchmark>/<model>/<method>.npy` (confidence, 0–1) and `y_true/...` (correctness). The data cover 11 models (Gemma 1.1 2B and 7B, GPT-3.5-turbo, GPT-4o, GPT-4o-mini, Llama 3 8B and 70B, Qwen1.5 7B, 32B, 72B and 110B), 10 benchmarks (ARC-Challenge, ARC-Easy, CommonsenseQA, LogiQA, MMLU, SciQ, Social IQa, TriviaQA, TruthfulQA MC1 and MC2) and 17 prompting methods. The data are not redistributed here; obtain them from the original authors.

- Primary analysis: their `basic_1s` prompt, 110 model × benchmark cells, 112,909 responses.
- Correctness for TriviaQA and TruthfulQA MC2 is fractional partial credit; we count ≥ 0.5 as correct (counting any credit leaves the verdicts at cut point 0.9 unchanged).

Results (validity-screen 0.3.0):

- At cut point 0.5: 2 Valid, 14 Indeterminate, 94 Invalid. At 0.9: 56 Valid, 19 Indeterminate, 35 Invalid.
- Qwen1.5-7B-chat is Invalid on all 10 benchmarks at 0.5 and 0.9: its confidence barely varies (mean within-cell SD .011), and AUROC averages .501.
- Gemma 1.1 7B is Invalid on all 10 benchmarks at 0.9 (L between .97 and 1.00, certified in every cell).
- Across the 17 prompting methods, the share of cells Invalid at 0.9 ranges from 19% to 60%: verdicts depend on the prompt as well as the model.
- Mean raw-confidence AUROC across cells: .607 (.43 to .89).
