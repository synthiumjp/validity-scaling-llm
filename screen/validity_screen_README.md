# validity-screen

**Validity screening protocol for LLM confidence signals.**

[![PyPI](https://img.shields.io/pypi/v/validity-screen)](https://pypi.org/project/validity-screen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implements the Stage A screening sequence from:

> Cacioli, J. P. (2026). *Screen Before You Interpret: A Portable Validity Protocol for Benchmark-Based LLM Confidence Signals.* arXiv.

## Install

```bash
pip install validity-screen
```

Or use directly from this repo:

```python
from validity_screen import screen
```

## What it does

Before computing calibration metrics (ECE), metacognitive sensitivity (meta-d', AUROC), or selective prediction accuracy on an LLM's confidence data, this protocol checks whether the confidence signal carries item-level information about correctness. If it doesn't, those downstream metrics are fitting noise.

The protocol computes five values from a 2×2 contingency table (correct/incorrect × high/low confidence) and classifies the signal as **Valid**, **Indeterminate**, or **Invalid**.

## Quick start (Python)

```python
import numpy as np
from validity_screen import screen

# Your data: item-level correctness and confidence
correct    = np.array([True, True, False, True, False, ...])   # did the model get it right?
confidence = np.array([True, True, True,  True, False, ...])   # did the model express high confidence?

result = screen(correct, confidence, model_name="My Model", benchmark_name="MMLU")

print(result.tier)          # 'Valid', 'Indeterminate', or 'Invalid'
print(result.vrs_table())   # Complete VRS Table for reporting
```

## Quick start (command line)

```bash
# From a CSV with 'correct' and 'confidence' columns
validity-screen run --data my_data.csv --model-name "GPT-5.4"

# Continuous confidence? Binarised at threshold or median
validity-screen run --data my_data.csv --confidence-col prob --threshold 0.5

# JSON output for pipelines
validity-screen run --data my_data.csv --json
```

## What the tiers mean

| Tier | Meaning | Action |
|------|---------|--------|
| **Valid** | Confidence tracks correctness at the item level | Proceed with downstream metrics |
| **Indeterminate** | Near threshold; classification uncertain | Compute metrics but flag them; consider more items |
| **Invalid** | Confidence does not discriminate correct from incorrect | Do not interpret AUROC, ECE, meta-d', selective prediction |

## Indices

| Index | Formula | What it detects | Threshold |
|-------|---------|-----------------|-----------|
| **L** | P(high conf \| incorrect) | Blanket confidence on errors | ≥ 0.95 → Invalid |
| **Fp** | P(low conf \| correct) | Over-withdrawal of correct items | ≥ 0.50 → Invalid |
| **RBS** | Fp − (1 − L) | Inverted monitoring direction | > 0 (CI excl. zero) → Invalid |
| **TRIN** | max(n_high, n_low) / N | Fixed responding | ≥ 0.95 → structural warning only |
| **r** | point-biserial(conf, correct) | Item-level sensitivity | Report value, p, CI |

## Continuous confidence

If your confidence signal is continuous (0–100, logit probabilities), binarise first:

```python
from validity_screen import screen, binarise

confidence_binary = binarise(confidence_continuous, threshold=50)
# or: binarise(confidence_continuous, method='median')

result = screen(correct, confidence_binary)
```

## Batch screening

```python
from validity_screen import screen_batch, summary_table

models = {
    "GPT-4o": {"correct": correct_gpt, "confidence": conf_gpt},
    "Claude": {"correct": correct_claude, "confidence": conf_claude},
}

results = screen_batch(models, benchmark_name="MMLU")
print(summary_table(results))
```

## VRS Table

The protocol requires a **Validity Report for Confidence Screening** (VRS Table) alongside any metacognitive, calibration, or selective prediction metric. The `vrs_table()` method generates this:

```python
result = screen(correct, confidence,
    model_name="Claude Haiku 4.5",
    benchmark_name="Classical Minds v1",
    elicitation_method="Binary probe (KEEP / WITHDRAW)",
    probe_timing="Retrospective"
)
print(result.vrs_table())
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7

## Citation

```bibtex
@article{cacioli2026screen,
  title={Screen Before You Interpret: A Portable Validity Protocol for 
         Benchmark-Based LLM Confidence Signals},
  author={Cacioli, Jon-Paul},
  journal={arXiv preprint},
  year={2026}
}
```

## Companion papers

The validity indices are derived and psychometrically validated in:

> Cacioli, J. P. (2026). *Before You Interpret the Profile: Validity Scaling for LLM Metacognitive Self-Report.* arXiv.

Concurrent criterion validation against selective prediction:

> Cacioli, J. P. (2026). *Concurrent Criterion Validation of a Validity Screen for LLM Confidence Signals via Selective Prediction.* arXiv.

## License

MIT
