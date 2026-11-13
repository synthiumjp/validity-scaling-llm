# Analysis code

- **`v2/`**: corrected analysis for arXiv:2604.17714v2. `run_all.sh` regenerates every number, table and figure; outputs are committed in `v2/out/` (primary, five keyed battery tracks) and `v2/out_sixtrack/` (all six tracks, sensitivity). `v2/validity_screen.py` is the analysis copy of the screen; it gives identical results to the `validity-screen` 0.3.0 package (`screen/`).
- **`*.py` in this folder**: version 1 scripts, superseded. They score declined responses as confident errors, use the substring-graded prospective track and hard-code some tiers, which produced the errors listed in the correction notice in the top-level README. Kept for the record only.
