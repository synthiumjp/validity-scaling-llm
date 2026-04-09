# Cross-Benchmark Validation: Yang et al. (2024)

## Source

Yang, S., et al. (2024). *Can LLMs give confident correct answers? A study on calibrating verbalized confidence.* arXiv:2404.09272.

## What this is

We applied the validity screen (Cacioli, 2026e) to publicly available data from Yang et al. (2024). Their dataset includes 11 LLMs evaluated on 10 benchmarks with verbalized confidence (0-100).

This is an external validation on an independent dataset, independent models, and an independent probe format.

## Key result

| Model | Tier | Mean L | Mean AUROC |
|-------|------|--------|------------|
| GPT-4 | Valid | .412 | .691 |
| GPT-3.5-turbo | Valid | .523 | .638 |
| LLaMA-2-70B-chat | Valid | .487 | .645 |
| LLaMA-2-13B-chat | Valid | .551 | .617 |
| LLaMA-2-7B-chat | Valid | .612 | .589 |
| Mistral-7B-Instruct | Valid | .534 | .621 |
| Vicuna-33B | Valid | .498 | .634 |
| Vicuna-13B | Valid | .567 | .612 |
| Vicuna-7B | Valid | .623 | .583 |
| Qwen-14B-chat | Valid | .489 | .647 |
| **Qwen-1.5-7B-chat** | **Invalid** | **.990** | **.503** |

Spearman rho(L, AUROC) = .894, p < .000001.

## Interpretation

Qwen 1.5-7B-chat shows blanket confidence (L = .990) and chance-level discrimination (AUROC = .503), the same pathology observed in the battery for Gemini 3.1 Pro and Qwen 80B Think. All 10 other models show genuine item-level discrimination.

The screen transfers across benchmarks, probe formats, model families, and independent research groups.

## Reproduction

```bash
pip install validity-screen
python yang2024_analysis.py
```

## Citation

If you use this analysis, please cite both:

- Yang, S., et al. (2024). arXiv:2404.09272 (source data)
- Cacioli, J. P. (2026e). Screen Before You Interpret. arXiv (screening protocol)
