# Cross-benchmark analyses (verbalized confidence)

Corrected in November 2026 for arXiv:2604.17714v2. The earlier version of this folder reported a Yang et al. (2024) reanalysis that was not computed from that study's data, and MMLU tiers from a median split whose verdict depends on how ties are handled. Both are withdrawn.

## What is here

- [`mmlu/`](mmlu/): 20 battery models on 500 stratified MMLU items with verbalized 0–100 confidence (item level).
- [`yang2024/`](yang2024/): notes on the analysis of Yang, Tsai and Yamada (2024), arXiv:2412.14737.

The 33-model atlas (1,500 MMLU items) lives in [synthiumjp/metacognitive-profile-atlas](https://github.com/synthiumjp/metacognitive-profile-atlas). All three data sets are analysed by `analysis/v2/mmlu_rescore.py` and `analysis/v2/yang_rescore.py`; run everything with `analysis/v2/run_all.sh`.

## Results (validity-screen 0.3.0)

Verbalized confidence piles at 90–100, so the verdict depends on the cut point c (confidence ≥ c counts as high):

| Data | Units | c = 50 (Valid / Indet. / Invalid) | c = 90 | c = 95 |
|---|---|---|---|---|
| Atlas MMLU | 33 models | 9 / 0 / 24 | 28 / 1 / 4 | 29 / 1 / 3 |
| MMLU-500 | 20 models | 4 / 7 / 9 | 14 / 3 / 3 | 15 / 1 / 4 |
| Yang et al. | 110 model × benchmark cells | 2 / 14 / 94 | 56 / 19 / 35 | 45 / 13 / 52 |

- A Valid verdict at a high cut point means only that the cut separates errors better than chance. Gemma 3 12B is Valid at 95 with a raw-confidence AUROC of .505. Choose one cut point in advance (ideally the deployment threshold) and report it.
- Raw-confidence AUROC correlates r = .88 across the two MMLU samples for the 20 shared models.
- Battery item sensitivity (binary probe) does not predict MMLU AUROC (ρ = −.07). DeepSeek-R1, Gemini 3.1 Pro and Qwen 80B Thinking, which are not Valid on the battery, are Valid at 90 on both MMLU samples. Gemma 3 1B, 12B and 27B, which are Valid on the battery, are Invalid at 50 and 90.
- In the atlas, 980 of GPT-oss-120B's 1,500 confidences are parser defaults (50) and are excluded.
