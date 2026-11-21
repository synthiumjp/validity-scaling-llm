# validity-screen

**Check whether an LLM's confidence signal carries information about correctness before you build anything on it.**

[![PyPI](https://img.shields.io/pypi/v/validity-screen)](https://pypi.org/project/validity-screen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Abstention, routing and escalation all assume that a model's confidence varies with whether it is right. This package checks that assumption, using the logic psychological assessment applies to response validity: exclude responses the respondent did not give, rule out fixed response sets, and require evidence that responding follows content.

## Install

```bash
pip install validity-screen
```

## Quick start

```python
from validity_screen import screen

# Binary confidence (1 = kept / high, 0 = withdrawn / low)
r = screen(correct, keep, answered=answered, model_name="Model A")

# Continuous confidence: state the cut point you will deploy at, on the same scale as the data
r = screen(correct, confidence_0_100, answered=answered, cut=90)

print(r.tier, r.reason_code, r.flags)   # e.g. Valid signal_detected []
print(r.vrs_table())                    # full validity report
```

`answered` marks rows to drop before scoring: declines, unparseable output, and values the pipeline supplied rather than the model (such as a parser default). Missing values on scored rows raise an error rather than being guessed; with `cut`, missing confidence values are excluded and counted.

## What it reports

From the correct/wrong × kept/withdrawn table:

| Index | Formula | Meaning |
|---|---|---|
| Kept-error rate `L` | b / (b + d) | share of errors on which confidence was kept |
| Withdrawn-correct rate `Fp` | c / (a + c) | share of correct answers withdrawn |
| Item sensitivity `J` | (1 − Fp) − L | Youden's J; binary AUROC = (1 + J) / 2 |

## Tiers (version 0.4)

| Tier | Rule |
|---|---|
| **Invalid** | blanket confidence (L ≥ .95) or blanket withdrawal (Fp ≥ .95), each certified by a Wilson lower bound > .90; or inversion (J < 0, one-sided mid-p < .025) |
| **Uncertifiable** | fewer than 10 errors or 10 correct answers, and no certified response set |
| **Indeterminate** | response set or inversion suspected but not certified, or no detectable sensitivity (`reason_code` says which) |
| **Valid** | J > 0 (one-sided Fisher mid-p < .025) and no response set |

Flags, reported beside the tier and never changing it: `low_coverage` (Fp ≥ .50), `response_set_not_excluded` (Valid, but the L or Fp 95% interval reaches .95), `fixed_responding` (one response on ≥ 95% of items).

**Valid means better than chance, not good.** Read J and its interval. With many items a very small J is Valid; use `min_j=` (chosen in advance) if you need a minimum effect.

## Operating characteristics

- **False Valid** (no real signal): at most .026 against a nominal .025 in typical cases (226 to 1,500 items, accuracy .70 to .97, keep rates .50 to .99). With only 10 to 12 errors it reaches about .030 at keep rates of .50 or more. At low keep rates (.10 to .30, as at very high cut points) it reaches about .037.
- **False Invalid**: a model whose true kept-error rate is .93 is certified as blanket confidence up to about 23% of the time, depending on the number of errors. The guard is set at a Wilson lower bound of .90.
- **Power** depends on the number of errors, not items. Use the planning functions:

```python
from validity_screen import power_valid, power_invalid, items_needed, errors_needed
power_valid(n_items=1000, accuracy=0.90, keep_correct=0.95, keep_wrong=0.85)  # P(Valid) for J = .10
items_needed(accuracy=0.90, true_L=0.97, power=0.90)                          # 1600
errors_needed(withdrawn=0)                                                     # 35
```

## Command line

```bash
validity-screen run --data results.csv --correct-col is_correct --confidence-col keep
validity-screen run --data results.csv --confidence-col confidence --cut 90
validity-screen plan --accuracy 0.92 --true-L 0.97
```

Cells are read strictly. Blank or unreadable cells, including words in a numeric column, are excluded and counted, never guessed.

## Changes

See [CHANGELOG.md](CHANGELOG.md). Results from 0.1 should be rerun. Version 0.4 changed the decision rules; the decision record, with the rationale for each rule, is in the repository.

## Citation

Cacioli, J. P. (2026). *Screen Before You Interpret: A Validity Screen for LLM Confidence Signals and Its Operating Characteristics.* Preprint: arXiv:2604.17714.

## License

MIT
