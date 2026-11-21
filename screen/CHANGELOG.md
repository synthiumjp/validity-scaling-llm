# Changelog

## 0.4.1 (November 2026)

Fixes found by an independent review of 0.4.0. Decision rules unchanged.

- Scale mismatch is an error: 0-1 confidence with a 0-100 cut point (or the reverse) used to certify a response set without warning.
- `cut` must be a finite number; `alpha` must be in (0, 1); inputs must be one-dimensional; `answered` must match in length.
- `items_needed` returned one item too many through floating-point rounding (351 instead of 350).
- `response_set_not_excluded` is computed from 95% intervals whatever `alpha` is.
- Planning functions and `validity-screen plan` validate their arguments.
- No scored rows gives reason `no_scored_rows`.
- New `min_j=` option: Valid additionally requires the lower 95% bound of J to exceed a stated minimum.
- CLI: percent strings are read; words in a numeric column are unreadable (excluded and counted) instead of becoming 0/1; a column with no readable cell is an error; header lines in `--correct/--confidence` files are detected or skipped with `--header`.
- 56 tests.

## 0.4.0 (November 2026)

Decision rules changed (rationale in the decision record).

- Blanket withdrawal is Invalid at Fp >= .95 (certified), symmetric with blanket confidence; Fp >= .50 is now the `low_coverage` flag. (0.3 called signals with strong discrimination Invalid.)
- Valid requires J > 0 by a one-sided Fisher mid-p test at .025; the 0.3 Newcombe-interval rule exceeded its nominal false-Valid rate (up to .088).
- Every Indeterminate verdict has a `reason_code`; Valid verdicts whose L or Fp interval reaches .95 carry `response_set_not_excluded`.
- `screen(..., cut=)` handles continuous confidence, excluding and counting missing values. `binarise` has no default threshold. Non-binary or missing confidence passed without `cut` raises an error.
- `errors_needed(withdrawn)` returns integer-attainable counts; new `power_invalid`, `power_valid`, `items_needed(accuracy, true_L, power)`.
- RBS is reported as J (Youden's J); `.rbs` remains as a deprecated alias.

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
