# Changelog

## 0.3.0 (November 2026)

Corrects version 0.1.0. Results produced with 0.1.0 should be rerun.

**Changed tier rules**
- **Valid now requires evidence of item sensitivity**: no flag fires *and* the RBS interval lies entirely below 0 (Youden's J significantly above 0). Otherwise the tier is Indeterminate. Under the old rule an uninformative random policy was classified Valid in up to half of simulated draws. `require_signal=False` restores the old rule; `tier_v02` is always reported.
- **Per-index denominator guards** replace the "any cell < 5 → insufficient data" guard, which blocked exactly the response sets the screen targets (blanket confidence empties cell d). Each index is computed when its own denominator is at least 10. New tier: **Uncertifiable** (RBS cannot be estimated).
- RBS is Invalid when it is positive and its interval excludes 0 (no extra point-estimate condition).

**Statistics**
- RBS interval is now the Newcombe (1998) hybrid score interval; the Wald interval used before collapses at L = 1 or Fp = 0.
- Results report Youden's J and binary AUROC = (1 - RBS)/2 explicitly.

**Inputs**
- New `answered` argument (CLI `--answered-col`): declines, unparseable outputs and pipeline-supplied values are excluded before scoring and counted (`n_excluded`). Version 0.1 analyses that scored declines as confident errors produced spurious response sets.
- `binarise(method="median")` is deprecated; the CLI no longer defaults to a median split and requires `--threshold` for continuous confidence.

**New**
- `errors_needed(L)` and `items_needed(accuracy, L)`; CLI `validity-screen plan`.

## 0.1.0 (April 2026)

Initial release.
