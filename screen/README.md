# validity-screen

**Check whether an LLM's confidence signal tracks correctness, and which response set dominates it, before you build on it.**

[![PyPI](https://img.shields.io/pypi/v/validity-screen)](https://pypi.org/project/validity-screen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implements the screen in:

> Cacioli, J. P. (2026). *Screen Before You Interpret: A Portable Validity Screen for Benchmark-Based LLM Confidence Signals.* arXiv:2604.17714v2.

> **Version 0.3.0 corrects version 0.1.** See [CHANGELOG.md](CHANGELOG.md). Results produced with 0.1 should be rerun.

## Install

```bash
pip install validity-screen
```

## Quick start

```python
import numpy as np
from validity_screen import screen, binarise

# item-level correctness and a binary confidence judgement (1 = KEEP / high)
result = screen(correct, keep, model_name="My Model")
print(result.tier)          # Valid, Indeterminate, Invalid or Uncertifiable
print(result.vrs_table())   # full reporting table

# verbalized 0-100 confidence: choose the cut point in advance
result = screen(correct, binarise(confidence, 90), binarisation_threshold=">= 90")

# exclude declines, parse failures and pipeline defaults before scoring
result = screen(correct, keep, answered=answered)
```

Command line:

```bash
validity-screen run --data results.csv --correct-col is_correct --confidence-col keep
validity-screen run --data results.csv --confidence-col confidence --threshold 90
validity-screen run --data results.csv --answered-col parsed --json
validity-screen plan --accuracy 0.92        # items needed to certify blanket confidence
```

## What it computes

From the correctness-by-confidence table (a = correct & high, b = incorrect & high, c = correct & low, d = incorrect & low):

| Index | Definition | Standard equivalent | Flag |
|---|---|---|---|
| **L** | b / (b + d): confidence kept on errors | false-positive rate of KEEP | >= .95 |
| **Fp** | c / (a + c): confidence withdrawn on correct answers | false-negative rate of KEEP | >= .50 |
| **RBS** | Fp - (1 - L) | minus Youden's J; binary AUROC = (1 - RBS)/2 | > 0 |
| **TRIN** | max(n_high, n_low) / N | modal-response share | >= .95, warning only |
| **r** | Pearson on 0/1 vectors | phi | reported |

Each index is computed only when its denominator reaches 10. A flagged index is **Invalid** if its interval clears the guard (Wilson lower bound > .40 for Fp, > .90 for L; Newcombe interval excluding 0 for RBS) and **Indeterminate** otherwise.

## Tiers

| Tier | Meaning |
|---|---|
| **Invalid** | A response set is established: blanket confidence, blanket withdrawal, or inversion. Do not interpret AUROC, ECE or selective prediction. |
| **Indeterminate** | A threshold is crossed but cannot be certified (often too few errors), or no response set is present but confidence does not measurably track correctness. |
| **Uncertifiable** | RBS cannot be estimated (fewer than 10 correct items or errors). |
| **Valid** | No response set, and the RBS interval lies below 0: confidence measurably tracks correctness. |

`require_signal=False` (CLI `--v02`) gives the 0.2 rule, under which Valid only meant that no flag fired. Every result also reports `tier_v02`.

## Things to know

- **Errors, not items, set the sample size.** Certifying blanket confidence needs 35 errors at L = 1.00, 71 at .97, 97 at .96 and 139 at .95. Use `errors_needed()` / `items_needed()`.
- **Continuous confidence needs a cut point chosen in advance.** Verdicts depend on it: verbalized confidence usually piles at 90-100, so at 50 most models look like blanket confidence. Screen at the threshold your deployment will use. Median splits are deprecated because their verdict depends on tie handling.
- **Valid is necessary, not sufficient.** It says the signal is non-degenerate and better than chance, not that it is good. Report r and AUROC alongside the tier.
- **Verdicts are specific to probe format, prompt and task.** Screen under the conditions the signal will be used in.

## Citation

```bibtex
@misc{cacioli2026screen,
  author = {Cacioli, Jon-Paul},
  title  = {Screen Before You Interpret: A Portable Validity Screen for Benchmark-Based {LLM} Confidence Signals},
  year   = {2026},
  howpublished = {arXiv:2604.17714v2}
}
```

## License

MIT.
