# Cross-Benchmark Validation: MMLU with Verbalized Confidence

## Design

18 models from the battery (Cacioli, 2026c) were evaluated on 500 stratified MMLU items using verbalized confidence (0-100), binarised at the median. Two models (DeepSeek V3.2 and Qwen Coder 480B) were excluded due to insufficient items from API instability.

The screening tool was installed via `pip install validity-screen` and run without modification on a Kaggle benchmark.

## Key results

| Model | Battery Tier | MMLU Tier | N | Accuracy | AUROC | r |
|-------|-------------|-----------|---|----------|-------|---|
| Claude Opus 4.6 | Valid | Valid | 500 | .930 | .836 | +.217 |
| Claude Sonnet 4.6 | Valid | Valid | 498 | .904 | .837 | +.250 |
| Claude Haiku 4.5 | Valid | Valid | 500 | .890 | .760 | +.283 |
| Gemini 2.5 Pro | Valid | Valid | 364 | .915 | .803 | +.558 |
| Gemini 2.5 Flash | Valid | Valid | 500 | .914 | .724 | +.246 |
| Gemini 3 Flash | Valid | Valid | 372 | .933 | .695 | +.201 |
| Gemini 3.1 Pro | **Invalid** | **Valid** | 386 | .930 | .786 | +.294 |
| GLM-5 | Valid | Valid | 186 | .930 | .679 | +.135 |
| GPT-5.4 | Valid | Valid | 500 | .888 | .715 | +.190 |
| GPT-5.4 mini | Valid | Valid | 500 | .820 | .699 | +.235 |
| GPT-5.4 nano | Indeterminate | Valid | 500 | .656 | .554 | +.066 |
| Gemma 3 27B | Valid | Valid | 500 | .782 | .588 | +.147 |
| Gemma 3 12B | Indeterminate | Valid | 500 | .722 | .533 | +.106 |
| Gemma 3 1B | Indeterminate | **Invalid** | 500 | .388 | .500 | -.000 |
| DeepSeek-R1 | **Invalid** | **Valid** | 387 | .933 | .768 | +.204 |
| Qwen 80B Think | **Invalid** | **Valid** | 500 | .906 | .718 | +.265 |
| Qwen 80B Instruct | Valid | Valid | 500 | .856 | .613 | +.141 |
| Qwen 235B | Valid | Valid | 500 | .858 | .647 | +.221 |

## Three findings

1. **The screen transfers.** All 11 battery-Valid models remained Valid on MMLU. Gemma 1B remained Invalid. The construct generalises across benchmarks.

2. **Probe format modulates classification.** All three battery-Invalid models (R1, Gemini 3.1 Pro, Qwen Think) shifted to Valid on verbalized confidence. Binary KEEP/WITHDRAW collapses a gradient that continuous 0-100 preserves.

3. **The binary probe is the harder test.** Passing the binary screen implies passing the continuous screen. The reverse is not true.

## Excluded models

- **DeepSeek V3.2**: 114 items (API instability). Classified "Insufficient data".
- **Qwen Coder 480B**: 154 items (API instability). r CI crosses zero.

## Files

- `summary/` — One screening summary CSV per model (18 files)
- `item_level/` — One item-level results CSV per model (16 files; R1 and Flash parsed from run.json)

## Reproduction

```bash
pip install validity-screen
# Run the Kaggle benchmark: https://www.kaggle.com/benchmarks/validity-screen-mmlu
```

## Citation

Cacioli, J. P. (2026e). Screen Before You Interpret: A Portable Validity Protocol for Benchmark-Based LLM Confidence Signals. arXiv.
