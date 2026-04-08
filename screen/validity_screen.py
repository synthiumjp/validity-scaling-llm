"""
screen_before_you_interpret
===========================

Validity screening protocol for LLM confidence signals.

Implements the Stage A screening sequence from:
    Cacioli, J. P. (2026). Screen Before You Interpret: A Portable Validity
    Protocol for Benchmark-Based LLM Confidence Signals. arXiv.

Usage:
    from validity_screen import screen, vrs_table

    result = screen(correct, confidence)
    print(result.tier)          # 'Valid', 'Indeterminate', or 'Invalid'
    print(result.vrs_table())   # Formatted VRS Table
"""

__version__ = "0.1.0"
__author__ = "Jon-Paul Cacioli"

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from scipy import stats as sp_stats


# ============================================================
# Wilson score confidence interval
# ============================================================

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    (lower, upper) : tuple of float
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ============================================================
# Data classes for results
# ============================================================

@dataclass
class IndexResult:
    """Result for a single validity index."""
    name: str
    value: float
    ci_lower: float
    ci_upper: float
    threshold: Optional[float] = None
    flag: str = "ok"       # 'ok', 'invalid', 'indeterminate', 'warning'
    note: str = ""


@dataclass
class ScreenResult:
    """Complete screening result for one model."""
    # Metadata
    model_name: str = ""
    benchmark_name: str = ""
    n_items: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    accuracy: float = 0.0
    elicitation_method: str = ""
    confidence_format: str = ""
    binarisation_threshold: str = "N/A"
    probe_timing: str = ""

    # 2x2 table
    a: int = 0  # correct + high confidence
    b: int = 0  # incorrect + high confidence
    c: int = 0  # correct + low confidence
    d: int = 0  # incorrect + low confidence

    # Index results
    trin: Optional[IndexResult] = None
    L: Optional[IndexResult] = None
    Fp: Optional[IndexResult] = None
    rbs: Optional[IndexResult] = None
    r_conf_correct: Optional[IndexResult] = None

    # Classification
    tier: str = ""  # 'Valid', 'Indeterminate', 'Invalid', 'Insufficient data'
    flagging_reasons: List[str] = field(default_factory=list)
    response_style: str = ""

    # Cell count warning
    min_cell: int = 0

    def vrs_table(self) -> str:
        """Return a formatted VRS Table string."""
        rows = [
            ("Model", self.model_name or "[not specified]"),
            ("Benchmark", self.benchmark_name or "[not specified]"),
            ("N items", str(self.n_items)),
            ("N correct / N incorrect", f"{self.n_correct} / {self.n_incorrect}"),
            ("Accuracy", f"{self.accuracy:.3f}"),
            ("Confidence elicitation", self.elicitation_method or "[not specified]"),
            ("Confidence format", self.confidence_format or "[not specified]"),
            ("Binarisation threshold", self.binarisation_threshold),
            ("Probe timing", self.probe_timing or "[not specified]"),
            ("2x2 table", f"a={self.a}, b={self.b}, c={self.c}, d={self.d}"),
        ]

        if self.trin:
            direction = "fixed-high" if self.a + self.b > self.c + self.d else "fixed-low"
            warn = " — structural warning" if self.trin.value >= 0.95 else ""
            rows.append(("TRIN", f"{self.trin.value:.3f} ({direction}){warn}"))

        if self.L:
            rows.append(("L", f"{self.L.value:.3f} [{self.L.ci_lower:.3f}, {self.L.ci_upper:.3f}]"))

        if self.Fp:
            rows.append(("Fp", f"{self.Fp.value:.3f} [{self.Fp.ci_lower:.3f}, {self.Fp.ci_upper:.3f}]"))

        if self.rbs:
            rows.append(("RBS", f"{self.rbs.value:+.3f} [{self.rbs.ci_lower:+.3f}, {self.rbs.ci_upper:+.3f}]"))

        if self.r_conf_correct:
            r = self.r_conf_correct
            sig = "p < .001" if r.ci_lower > 0 or r.ci_upper < 0 else f"p = {r.threshold:.3f}" if r.threshold else ""
            rows.append(("r(confidence, correct)",
                         f"{r.value:+.3f}, {sig}, 95% CI [{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]"))

        rows.append(("Tier classification", self.tier))
        rows.append(("Flagging reason", "; ".join(self.flagging_reasons) if self.flagging_reasons else "None"))
        if self.response_style:
            rows.append(("Response style", self.response_style))

        # Format
        max_label = max(len(r[0]) for r in rows)
        lines = []
        lines.append("=" * (max_label + 50))
        lines.append("VRS TABLE — Validity Report for Confidence Screening")
        lines.append("=" * (max_label + 50))
        for label, value in rows:
            lines.append(f"  {label:<{max_label}}  {value}")
        lines.append("=" * (max_label + 50))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a dictionary of all fields for serialisation."""
        d = {
            "model_name": self.model_name,
            "benchmark_name": self.benchmark_name,
            "n_items": self.n_items,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "accuracy": round(self.accuracy, 4),
            "cell_a": self.a, "cell_b": self.b,
            "cell_c": self.c, "cell_d": self.d,
            "min_cell": self.min_cell,
            "tier": self.tier,
            "flagging_reasons": self.flagging_reasons,
            "response_style": self.response_style,
        }
        for idx_name in ["trin", "L", "Fp", "rbs", "r_conf_correct"]:
            idx = getattr(self, idx_name)
            if idx:
                d[f"{idx_name}_value"] = round(idx.value, 4)
                d[f"{idx_name}_ci_lower"] = round(idx.ci_lower, 4)
                d[f"{idx_name}_ci_upper"] = round(idx.ci_upper, 4)
                d[f"{idx_name}_flag"] = idx.flag
        return d


# ============================================================
# Main screening function
# ============================================================

def screen(
    correct: np.ndarray,
    confidence: np.ndarray,
    model_name: str = "",
    benchmark_name: str = "",
    elicitation_method: str = "",
    confidence_format: str = "",
    binarisation_threshold: str = "N/A",
    probe_timing: str = "",
    alpha: float = 0.05,
) -> ScreenResult:
    """Run the Stage A validity screening protocol.

    Parameters
    ----------
    correct : array-like of bool or 0/1
        Whether each item was answered correctly.
    confidence : array-like of bool or 0/1
        Whether the model expressed high confidence on each item.
        1 = high confidence (KEEP / BET), 0 = low confidence (WITHDRAW / NO BET).
    model_name : str, optional
        Model identifier for the VRS Table.
    benchmark_name : str, optional
        Benchmark identifier for the VRS Table.
    elicitation_method : str, optional
        How confidence was elicited.
    confidence_format : str, optional
        Original format before binarisation.
    binarisation_threshold : str, optional
        Threshold used if applicable.
    probe_timing : str, optional
        Retrospective, prospective, concurrent.
    alpha : float
        Significance level for confidence intervals (default 0.05).

    Returns
    -------
    ScreenResult
        Complete screening result including tier classification and VRS Table.
    """
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=bool)

    if len(correct) != len(confidence):
        raise ValueError(f"correct ({len(correct)}) and confidence ({len(confidence)}) must have same length")

    n = len(correct)
    n_correct = int(correct.sum())
    n_incorrect = n - n_correct

    # Build 2x2 table
    a = int((correct & confidence).sum())       # correct + high conf
    b = int((~correct & confidence).sum())      # incorrect + high conf
    c = int((correct & ~confidence).sum())      # correct + low conf
    d = int((~correct & ~confidence).sum())     # incorrect + low conf

    result = ScreenResult(
        model_name=model_name,
        benchmark_name=benchmark_name,
        n_items=n,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        accuracy=n_correct / n if n > 0 else 0.0,
        elicitation_method=elicitation_method,
        confidence_format=confidence_format,
        binarisation_threshold=binarisation_threshold,
        probe_timing=probe_timing,
        a=a, b=b, c=c, d=d,
        min_cell=min(a, b, c, d),
    )

    flags = []

    # ----------------------------------------------------------
    # Step 1: Check cell counts
    # ----------------------------------------------------------
    if min(a, b, c, d) < 5:
        result.tier = "Insufficient data"
        result.flagging_reasons = [f"Cell count below 5 (min cell = {min(a, b, c, d)})"]
        return result

    # ----------------------------------------------------------
    # Step 2: TRIN (structural indicator, not a Tier 1 flag)
    # ----------------------------------------------------------
    n_high = a + b
    n_low = c + d
    trin_val = max(n_high, n_low) / n
    trin_flag = "warning" if trin_val >= 0.95 else "ok"
    result.trin = IndexResult(
        name="TRIN", value=trin_val,
        ci_lower=trin_val, ci_upper=trin_val,  # deterministic
        threshold=0.95, flag=trin_flag,
        note="Structural warning only; does not trigger Invalid"
    )

    # ----------------------------------------------------------
    # Step 3: Fp = P(low confidence | correct)
    # ----------------------------------------------------------
    fp_val = c / n_correct if n_correct > 0 else 0.0
    fp_lo, fp_hi = wilson_ci(c, n_correct, alpha)
    fp_flag = "ok"
    if fp_val >= 0.50 and fp_lo > 0.40:
        fp_flag = "invalid"
        flags.append(f"Fp = {fp_val:.3f} exceeds .50 (Wilson CI lower = {fp_lo:.3f} > .40)")
    elif fp_val >= 0.50:
        fp_flag = "indeterminate"
        flags.append(f"Fp = {fp_val:.3f} at .50 but Wilson CI lower = {fp_lo:.3f} spans .40")
    result.Fp = IndexResult(name="Fp", value=fp_val, ci_lower=fp_lo, ci_upper=fp_hi,
                            threshold=0.50, flag=fp_flag)

    # ----------------------------------------------------------
    # Step 4: L = P(high confidence | incorrect)
    # ----------------------------------------------------------
    l_val = b / n_incorrect if n_incorrect > 0 else 0.0
    l_lo, l_hi = wilson_ci(b, n_incorrect, alpha)
    l_flag = "ok"
    if l_val >= 0.95 and l_lo > 0.90:
        l_flag = "invalid"
        flags.append(f"L = {l_val:.3f} exceeds .95 (Wilson CI lower = {l_lo:.3f} > .90)")
    elif l_val >= 0.95:
        l_flag = "indeterminate"
        flags.append(f"L = {l_val:.3f} at .95 but Wilson CI lower = {l_lo:.3f} spans .90")
    result.L = IndexResult(name="L", value=l_val, ci_lower=l_lo, ci_upper=l_hi,
                           threshold=0.95, flag=l_flag)

    # ----------------------------------------------------------
    # Step 5: RBS = Fp - (1 - L)
    # ----------------------------------------------------------
    rbs_val = fp_val - (1 - l_val)
    # CI for RBS via component SEs
    se_fp = np.sqrt(fp_val * (1 - fp_val) / n_correct) if n_correct > 0 else 0
    se_l = np.sqrt(l_val * (1 - l_val) / n_incorrect) if n_incorrect > 0 else 0
    se_rbs = np.sqrt(se_fp**2 + se_l**2)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    rbs_lo = rbs_val - z * se_rbs
    rbs_hi = rbs_val + z * se_rbs
    rbs_flag = "ok"
    if rbs_val > 0:
        if rbs_lo > 0:
            rbs_flag = "invalid"
            flags.append(f"RBS = {rbs_val:+.3f}, CI [{rbs_lo:+.3f}, {rbs_hi:+.3f}] excludes zero")
        else:
            rbs_flag = "indeterminate"
            flags.append(f"RBS = {rbs_val:+.3f}, CI [{rbs_lo:+.3f}, {rbs_hi:+.3f}] includes zero")
    result.rbs = IndexResult(name="RBS", value=rbs_val, ci_lower=rbs_lo, ci_upper=rbs_hi,
                             threshold=0.0, flag=rbs_flag)

    # ----------------------------------------------------------
    # Step 6: r(confidence, correct) — point-biserial
    # ----------------------------------------------------------
    conf_int = confidence.astype(int)
    corr_int = correct.astype(int)
    r_val, p_val = sp_stats.pointbiserialr(conf_int, corr_int)
    # Fisher z transform for CI
    z_r = np.arctanh(r_val) if abs(r_val) < 1.0 else np.sign(r_val) * 3.0
    se_z = 1.0 / np.sqrt(n - 3) if n > 3 else 1.0
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    r_lo = np.tanh(z_r - z_crit * se_z)
    r_hi = np.tanh(z_r + z_crit * se_z)
    result.r_conf_correct = IndexResult(
        name="r(confidence, correct)", value=r_val,
        ci_lower=r_lo, ci_upper=r_hi,
        threshold=p_val, flag="ok"
    )

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    has_invalid = any(idx_flag == "invalid" for idx_flag in
                      [result.Fp.flag, result.L.flag, result.rbs.flag])
    has_indet = any(idx_flag == "indeterminate" for idx_flag in
                    [result.Fp.flag, result.L.flag, result.rbs.flag])

    if has_invalid:
        result.tier = "Invalid"
    elif has_indet:
        result.tier = "Indeterminate"
    else:
        result.tier = "Valid"

    result.flagging_reasons = flags if flags else []

    # Response style characterisation for non-Valid
    if result.tier != "Valid":
        styles = []
        if result.L and result.L.value >= 0.95:
            styles.append(f"blanket confidence on errors (L = {result.L.value:.3f})")
        if result.Fp and result.Fp.value >= 0.50:
            styles.append(f"excessive withdrawal of correct items (Fp = {result.Fp.value:.3f})")
        if result.rbs and result.rbs.value > 0:
            styles.append(f"inverted monitoring (RBS = {result.rbs.value:+.3f})")
        if result.trin and result.trin.flag == "warning":
            styles.append(f"near-total response dominance (TRIN = {result.trin.value:.3f})")
        result.response_style = "; ".join(styles) if styles else "Near-threshold values"

    return result


# ============================================================
# Batch screening
# ============================================================

def screen_batch(
    models: dict,
    benchmark_name: str = "",
    **kwargs,
) -> List[ScreenResult]:
    """Screen multiple models.

    Parameters
    ----------
    models : dict
        Keys are model names, values are dicts with 'correct' and 'confidence' arrays.
    benchmark_name : str, optional
        Benchmark name for all models.

    Returns
    -------
    list of ScreenResult
    """
    results = []
    for name, data in models.items():
        r = screen(
            correct=data["correct"],
            confidence=data["confidence"],
            model_name=name,
            benchmark_name=benchmark_name,
            **kwargs,
        )
        results.append(r)
    return results


def summary_table(results: List[ScreenResult]) -> str:
    """Print a summary table of screening results."""
    header = f"{'Model':<25} {'Tier':<15} {'L':>6} {'Fp':>6} {'RBS':>7} {'TRIN':>6} {'r':>7}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in sorted(results, key=lambda x: x.tier):
        l_str = f"{r.L.value:.3f}" if r.L else "—"
        fp_str = f"{r.Fp.value:.3f}" if r.Fp else "—"
        rbs_str = f"{r.rbs.value:+.3f}" if r.rbs else "—"
        trin_str = f"{r.trin.value:.3f}" if r.trin else "—"
        r_str = f"{r.r_conf_correct.value:+.3f}" if r.r_conf_correct else "—"
        lines.append(f"{r.model_name:<25} {r.tier:<15} {l_str:>6} {fp_str:>6} {rbs_str:>7} {trin_str:>6} {r_str:>7}")
    lines.append(sep)
    return "\n".join(lines)


# ============================================================
# Convenience: binarise continuous confidence
# ============================================================

def binarise(confidence: np.ndarray, threshold: float = 0.5, method: str = "fixed") -> np.ndarray:
    """Binarise continuous confidence to high/low.

    Parameters
    ----------
    confidence : array-like
        Continuous confidence values.
    threshold : float
        Threshold value (default 0.5).
    method : str
        'fixed' uses the threshold directly.
        'median' uses the sample median.

    Returns
    -------
    np.ndarray of bool
        True = high confidence, False = low confidence.
    """
    confidence = np.asarray(confidence, dtype=float)
    if method == "median":
        threshold = np.median(confidence)
    return confidence >= threshold
