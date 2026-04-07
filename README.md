# Before You Interpret the Profile: Validity Scaling for LLM Metacognitive Self-Report

**Author:** Jon-Paul Cacioli  
**Programme:** Classical Minds, Modern Machines  
**Status:** Manuscript complete. arXiv submission pending (post April 16, 2026).

## Overview

This repository contains the data, analysis code, and manuscript materials for a methods paper applying clinical assessment validity scaling (PAI/MMPI-3) to LLM metacognitive probe data.

We operationalise six validity indices — L, K, F, Fp, RBS, and TRIN — mapped from the PAI (Morey, 1991, 2007) and MMPI-3 (Ben-Porath & Tellegen, 2020), and apply them to dual-probe (KEEP/WITHDRAW + BET/NO BET) data from 20 frontier LLMs across 524 items and six cognitive domains.

**Key findings:**
- A tiered classification system identifies 4 models as construct-level invalid and 2 as elevated
- Valid-profile models produce item-sensitive confidence (mean r = .18); invalid-profile models do not (mean r = -.20; d = 2.17, p = .001)
- Chain-of-thought training produces two opposite response distortions
- The WITHDRAW+BET contradiction rate is 37% in DeepSeek-R1, 0% in all other models
- Two latent dimensions (under-reporting, over-reporting) account for 94.6% of index variance

## Repository Structure

```
validity-scaling-llm/
├── README.md
├── data/
│   └── csvs/                    # 120 source CSVs (20 models × 6 tracks)
├── analysis/
│   ├── compute_indices.py       # Main analysis pipeline
│   ├── synthetic_validation.py  # Synthetic policy baseline
│   ├── robustness.py            # Leave-one-out, bootstrap, threshold sweep
│   └── figures.py               # Figure generation
├── figures/                     # Publication-quality figures
├── manuscript/
│   ├── validity_scaling_draft.md
│   └── validity_scaling_outline_v2.md
└── LICENSE
```

## Data

120 CSVs from the Classical Minds metacognitive monitoring battery (Cacioli, 2026). Each CSV contains per-item responses for one model on one track, including:
- `correct`: whether the model's answer was correct
- `keep_withdraw`: KEEP or WITHDRAW probe response
- `bet_nobet`: BET or NO BET probe response
- Track-specific metadata (item type, domain, difficulty, etc.)

**Source:** Cacioli, J. P. (2026). Classical Minds, Modern Machines: A cross-domain metacognitive monitoring benchmark for LLMs.

## Dependencies

```
python >= 3.10
pandas
numpy
scipy
```

## Citation

```bibtex
@misc{cacioli2026validity,
  title={Before You Interpret the Profile: Validity Scaling for LLM Metacognitive Self-Report},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={arXiv preprint (forthcoming)}
}
```

## Related Papers

- Cacioli (2026a). LLMs as signal detectors. arXiv:2603.14893
- Cacioli (2026b). Do LLMs know what they know? arXiv:2603.25112
- Cacioli (2026c). Classical Minds metacognitive battery. Manuscript in preparation.
- Cacioli (2026d). Weber's Law in transformer magnitude representations. arXiv:2603.20642
- Cacioli (2026e). Categorical perception in LLM hidden states. arXiv:2603.28258

## License

MIT
