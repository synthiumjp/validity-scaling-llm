# Validity Scaling for LLM Metacognitive Self-Report

**Author:** Jon-Paul Cacioli
**Programme:** Classical Minds, Modern Machines
**Status:** Three manuscripts complete. arXiv submission pending (post April 16, 2026).

## Overview

This repository contains the data, analysis code, and manuscript materials for three companion papers applying clinical assessment validity scaling (PAI/MMPI-3) to LLM metacognitive probe data.

We operationalise six validity indices mapped from the PAI (Morey, 1991, 2007) and MMPI-3 (Ben-Porath & Tellegen, 2020). We apply them to dual-probe (KEEP/WITHDRAW + BET/NO BET) data from 20 frontier LLMs across 524 items and six cognitive domains. We then validate the resulting classifications against selective prediction performance.

## The three papers

### Paper 1: Before You Interpret the Profile (Cacioli, 2026d)

Derivation study. Six validity indices (L, K, F, Fp, RBS, TRIN). Psychometric properties, factor structure, chain-of-thought training effects, the WITHDRAW+BET contradiction.

Key findings:
- A tiered classification system identifies 4 models as construct-level invalid and 2 as elevated
- Valid-profile models produce item-sensitive confidence (mean r = .18). Invalid-profile models do not (mean r = −.20, d = 2.17, p = .001)
- Chain-of-thought training produces two opposite response distortions
- The WITHDRAW+BET contradiction rate is 37% in DeepSeek-R1, 0% in all other models
- Two latent dimensions (under-reporting, over-reporting) account for 94.6% of index variance

### Paper 2: Screen Before You Interpret (Cacioli, 2026e)

Portable protocol. Three core indices (L, Fp, RBS), one structural indicator (TRIN), one diagnostic statistic (r(confidence, correct)). Three-tier classification (Invalid, Indeterminate, Valid). VRS Table reporting standard.

Key findings:
- Minimal screening protocol requires only a 2x2 contingency table
- Subsampling analysis shows stable classification at 100-150 items for clear cases
- Unscreened invalid models produce AUROC at chance and zero selective prediction gain

### Paper 3: Concurrent Criterion Validation via Selective Prediction (Cacioli, 2026f)

Predictive validation. Tests whether the tier classifications from the screen predict selective prediction performance.

Key findings:
- Valid models show mean Type 2 AUROC = .624. Invalid models show mean AUROC = .357. d = 2.81, p = .002
- Tiers order monotonically. Invalid < Indeterminate < Valid
- Split-half cross-validation yields median d = 1.77, P(d > 0) = 1.0 across 1,000 splits
- The three-tier classification accounts for 47% of the variance in AUROC (η² = .470)
- DeepSeek-R1 drops from 85.3% accuracy at full coverage to 11.3% at 10% coverage

## Repository Structure

```
validity-scaling-llm/
├── README.md
├── data/
│   └── csvs/                              # 120 source CSVs (20 models × 6 tracks)
│       ├── Attention/
│       ├── Executive/
│       ├── Meta Cog/
│       ├── Overhypothesis/
│       ├── Social Cognition/
│       └── prospective/
├── screen/
│   └── validity_screen.py                 # Portable screen implementation (Paper 2)
├── analysis/
│   ├── compute_indices.py                 # Validity index computation (Paper 1)
│   ├── synthetic_validation.py            # Synthetic policy baselines (Paper 1)
│   ├── robustness.py                      # Leave-one-out, bootstrap, threshold sweep (Paper 1)
│   ├── selective_prediction_analysis.py   # Selective prediction pipeline (Paper 3)
│   └── figures/                           # Figure generation scripts
├── papers/
│   ├── validity_scaling/
│   │   ├── validity_scaling_draft.md      # Paper 1 manuscript
│   │   └── figures/
│   ├── screen_protocol/
│   │   ├── screen_before_you_interpret.md # Paper 2 manuscript
│   │   └── figures/
│   └── selective_prediction/
│       ├── selective_prediction_draft.md   # Paper 3 manuscript
│       └── figures/
├── results/                               # Precomputed results tables
└── LICENSE
```

## Data

120 CSVs from the Classical Minds metacognitive monitoring battery (Cacioli, 2026c). Each CSV contains per-item responses for one model on one track, including:
- `correct` / `is_correct`: whether the model's answer was correct
- `keep_withdraw`: KEEP or WITHDRAW probe response
- `bet_nobet`: BET or NO BET probe response
- Track-specific metadata (item type, domain, difficulty, etc.)

Source: Cacioli, J. P. (2026c). Classical Minds, Modern Machines: A cross-domain metacognitive monitoring benchmark for LLMs. [NeurIPS E&D submission].

## Quick start

### Run the validity screen on your own data

```python
from screen.validity_screen import screen
import numpy as np

correct = np.array([True, True, False, True, False, ...])
confidence = np.array([True, True, True, False, False, ...])  # True = high confidence

result = screen(correct, confidence, model_name="my_model")
print(result.tier)          # 'Valid', 'Indeterminate', or 'Invalid'
print(result.vrs_table())   # Formatted VRS Table
```

### Run the selective prediction analysis

```bash
python analysis/selective_prediction_analysis.py
```

## Dependencies

```
python >= 3.10
pandas
numpy
scipy
scikit-learn
```

## The 20 models

| Model | Family | Tier | AUROC |
|-------|--------|------|-------|
| Sonnet 4.6 | Anthropic | Valid | .717 |
| Qwen Coder 480B | Qwen | Valid | .686 |
| Claude Haiku 4.5 | Anthropic | Valid | .657 |
| DeepSeek V3.2 | DeepSeek | Valid | .651 |
| Qwen 235B | Qwen | Valid | .648 |
| GPT-5.4 | OpenAI | Valid | .646 |
| GPT-5.4 mini | OpenAI | Valid | .633 |
| Gemma 3 27B | Google | Valid | .631 |
| Opus 4.6 | Anthropic | Valid | .617 |
| GLM-5 | Zhipu | Valid | .587 |
| Qwen 80B Inst | Qwen | Valid | .584 |
| Gemini 2.5 Flash | Google | Valid | .579 |
| Gemini 2.5 Pro | Google | Valid | .561 |
| Gemini 3 Flash | Google | Valid | .539 |
| Gemma 3 12B | Google | Indeterminate | .615 |
| GPT-5.4 nano | OpenAI | Indeterminate | .565 |
| Gemma 3 1B | Google | Indeterminate | .483 |
| Gemini 3.1 Pro | Google | Invalid | .522 |
| Qwen 80B Think | Qwen | Invalid | .518 |
| DeepSeek-R1 | DeepSeek | Invalid | .031 |

## Citation

```bibtex
@misc{cacioli2026validity,
  title={Before You Interpret the Profile: Validity Scaling for LLM Metacognitive Self-Report},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={arXiv preprint (forthcoming)}
}

@misc{cacioli2026screen,
  title={Screen Before You Interpret: A Portable Validity Protocol for Benchmark-Based LLM Confidence Signals},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={arXiv preprint (forthcoming)}
}

@misc{cacioli2026selective,
  title={Concurrent Criterion Validation of a Validity Screen for LLM Confidence Signals via Selective Prediction},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={arXiv preprint (forthcoming)}
}
```

## Related papers

- Cacioli (2026a). [LLMs as signal detectors](https://arxiv.org/abs/2603.14893). arXiv:2603.14893
- Cacioli (2026b). [Do LLMs know what they know?](https://arxiv.org/abs/2603.25112) arXiv:2603.25112
- Cacioli (2026c). Classical Minds metacognitive battery. [NeurIPS E&D submission]
- Cacioli (2026). [Overhypothesis formation in LLMs](https://arxiv.org/abs/2603.13696). arXiv:2603.13696
- Cacioli (2026). [Weber's Law in transformer magnitude representations](https://arxiv.org/abs/2603.20642). arXiv:2603.20642
- Cacioli (2026). [Categorical perception in LLM hidden states](https://arxiv.org/abs/2603.28258). arXiv:2603.28258

## License

MIT
