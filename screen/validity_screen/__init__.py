"""
validity-screen
===============

Validity screen for LLM confidence signals: checks whether a model's confidence
measurably tracks correctness, and whether a fixed response set (blanket
confidence, blanket withdrawal) or inversion makes it uninterpretable, before
downstream metrics (AUROC, ECE, meta-d', selective prediction) are computed.

Quick start::

    from validity_screen import screen, binarise

    result = screen(correct, keep, model_name="My Model")          # binary confidence
    result = screen(correct, conf_0_100, cut=90, model_name="My Model")  # continuous
    print(result.tier, result.reason, result.flags)
    print(result.vrs_table())

Reference:
    Cacioli, J. P. (2026). Screen Before You Interpret: A Portable Validity Screen
    for Benchmark-Based LLM Confidence Signals. arXiv:2604.17714v2.
"""

from validity_screen.core import (
    screen,
    screen_batch,
    summary_table,
    binarise,
    wilson_ci,
    newcombe_diff_ci,
    midp_signal,
    errors_needed,
    power_invalid,
    power_valid,
    items_needed,
    ScreenResult,
    IndexResult,
    MIN_N,
)

__version__ = "0.4.1"
__author__ = "Jon-Paul Cacioli"

__all__ = [
    "screen", "screen_batch", "summary_table", "binarise", "wilson_ci", "newcombe_diff_ci",
    "midp_signal", "errors_needed", "power_invalid", "power_valid", "items_needed",
    "ScreenResult", "IndexResult", "MIN_N",
]
