# MMLU with verbalized confidence (MMLU-500)

20 battery models answered 500 stratified MMLU items, stating a confidence from 0 to 100. Seven runs are partial (114 to 498 items) because of API instability.

**Corrected in November 2026.** The earlier version of this README reported tiers from a per-model median split. With confidence piled on a few values, that verdict depends on whether ties at the median count as high or low (sending them low turns 19 of 20 Valid models Invalid), so those tiers are withdrawn. It also listed DeepSeek-R1 and Gemini 2.5 Flash as summary-only; their item-level files, and those for DeepSeek V3.2 and Qwen Coder 480B, are now included.

## Files

- `item_level/`: one CSV per model (20 files): `item_id, subject, question, answer, correct_answer, is_correct, confidence, model`. `answer = X` marks an unparsed answer (6 rows), excluded before scoring.
- `summary/`: per-model summaries produced by the original (v0.1) benchmark run. Superseded; kept for the record.

## Corrected results

Produced by `analysis/v2/mmlu_rescore.py` (see `analysis/v2/out/mmlu20/mmlu_per_model.csv`).

- Raw-confidence AUROC ranges from .500 (Gemma 3 1B) to .837. The median model gives 90 or more on 93% of items.
- Tiers by cut point (Valid / Indeterminate / Invalid): c = 50: 4 / 7 / 9; c = 90: 14 / 3 / 3; c = 95: 15 / 1 / 4.
- Against the 33-model atlas (separate 1,500-item sample, 71 shared questions), AUROC correlates r = .88 for the 20 shared models. Tier agreement is 65% at c = 50 and 90% at c = 95.
