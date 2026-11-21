"""
validity_screen.core
====================

Validity screen for binary (or binarised) LLM confidence signals.

The screen works on four cells: correct and kept (a), wrong and kept (b),
correct and withdrawn (c), wrong and withdrawn (d). It reports

  kept-error rate          L  = b / (b + d)
  withdrawn-correct rate   Fp = c / (a + c)
  item sensitivity         J  = (1 - Fp) - L      (Youden's J)

Tier rules (version 0.4)
------------------------
Response sets are judged the same way in both directions. At L >= .95 or
Fp >= .95, J can be at most .05.

  Invalid        blanket confidence (L >= .95) or blanket withdrawal (Fp >= .95),
                 each certified by its Wilson lower bound (> .90); or inversion
                 (J < 0, one-sided mid-p < .025).
  Uncertifiable  fewer than 10 errors or 10 correct answers, and no certified
                 response set.
  Indeterminate  a response set or inversion suspected but not certified, or no
                 detectable item sensitivity. `reason_code` says which.
  Valid          J > 0 (one-sided Fisher mid-p < .025) and no response set.

Flags (reported beside the tier, never change it):
  low_coverage                 Fp >= .50: withdraws at least half its correct answers.
  response_set_not_excluded    Valid, but the L or Fp interval reaches .95.
  fixed_responding             TRIN >= .95: one response on at least 95% of items.

Decision record with clinical and statistical rationale: see the package README
and Cacioli, "Screen Before You Interpret" (arXiv:2604.17714).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

MIN_N = 10
RESPONSE_SET_CUT = 0.95      # L or Fp at or above this is a fixed response set
GUARD = 0.90                 # Wilson lower bound must exceed this to certify it
LOW_COVERAGE_CUT = 0.50
TRIN_CUT = 0.95
SIGNAL_ALPHA = 0.025         # one-sided level for J > 0 and for inversion

# Backward-compatible names (0.3)
L_CUT, L_GUARD = RESPONSE_SET_CUT, GUARD
FP_CUT, FP_GUARD = RESPONSE_SET_CUT, GUARD

REASON_TEXT = {
    "blanket_confidence": "blanket confidence: keeps at least 95% of its errors (certified)",
    "blanket_withdrawal": "blanket withdrawal: withdraws at least 95% of its correct answers (certified)",
    "inversion": "inversion: keeps errors more often than correct answers (significant)",
    "too_few_errors": "fewer than 10 errors: item sensitivity cannot be assessed",
    "too_few_correct": "fewer than 10 correct answers: item sensitivity cannot be assessed",
    "blanket_confidence_suspected": "L >= .95 but too few errors to certify blanket confidence",
    "blanket_withdrawal_suspected": "Fp >= .95 but too few correct answers to certify blanket withdrawal",
    "inversion_suspected": "J < 0 but not significant",
    "no_detectable_sensitivity": "no detectable item sensitivity: J > 0 not significant",
    "signal_detected": "J > 0 significant and no response set",
    "no_scored_rows": "no scored rows (all items excluded or none given)",
    "sensitivity_below_minimum": "J > 0 significant but its interval does not exceed the stated minimum",
}


# ============================================================
# Intervals and tests
# ============================================================

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int,
                     alpha: float = 0.05) -> Tuple[float, float]:
    """Newcombe (1998) hybrid score interval for p1 - p2 (independent samples)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, alpha)
    l2, u2 = wilson_ci(k2, n2, alpha)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (float(lo), float(hi))


def midp_signal(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    """One-sided Fisher mid-p values for the 2x2 table, conditional on the margins.

    Returns (p_signal, p_inversion): p_signal tests J > 0 (correct answers kept more
    often than errors); p_inversion tests J < 0. Mid-p = P(more extreme) + 0.5 P(observed).
    """
    nc, N, K = a + c, a + b + c + d, a + b
    pmf = sp_stats.hypergeom.pmf(a, N, nc, K)
    p_signal = sp_stats.hypergeom.sf(a, N, nc, K) + 0.5 * pmf
    p_inversion = sp_stats.hypergeom.cdf(a - 1, N, nc, K) + 0.5 * pmf
    return float(p_signal), float(p_inversion)


# ============================================================
# Result containers
# ============================================================

@dataclass
class IndexResult:
    """One index with its interval. `flag` is 'ok', 'certified', 'suspected',
    'not_estimable' or 'warning'."""
    name: str
    value: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    n: int = 0
    flag: str = "ok"
    note: str = ""

    @property
    def lo(self) -> float:
        return self.ci_lower

    @property
    def hi(self) -> float:
        return self.ci_upper


@dataclass
class ScreenResult:
    """Complete screening result for one model."""
    model_name: str = ""
    benchmark_name: str = ""
    elicitation_method: str = ""
    confidence_format: str = ""
    binarisation_threshold: str = "N/A"
    probe_timing: str = ""
    n_items: int = 0
    n_excluded: int = 0          # all rows not scored
    n_missing: int = 0           # of which: missing confidence or correctness
    n_scored: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    accuracy: float = float("nan")
    a: int = 0   # correct, kept
    b: int = 0   # wrong, kept
    c: int = 0   # correct, withdrawn
    d: int = 0   # wrong, withdrawn
    trin: Optional[IndexResult] = None
    L: Optional[IndexResult] = None
    Fp: Optional[IndexResult] = None
    J: Optional[IndexResult] = None
    r_conf_correct: Optional[IndexResult] = None
    p_signal: float = float("nan")
    p_inversion: float = float("nan")
    auroc_binary: float = float("nan")
    tier: str = ""
    reason_code: str = ""
    flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # ---- compatibility with 0.3 ----
    @property
    def youden_j(self) -> float:
        return self.J.value if self.J is not None else float("nan")

    @property
    def rbs(self) -> Optional[IndexResult]:
        """Deprecated: -J with the interval reversed (the 0.3 'RBS')."""
        if self.J is None:
            return None
        return IndexResult("RBS", -self.J.value, -self.J.ci_upper, -self.J.ci_lower, self.J.n,
                           self.J.flag, self.J.note)

    @property
    def r(self) -> Optional[IndexResult]:
        return self.r_conf_correct

    @property
    def reason(self) -> str:
        return REASON_TEXT.get(self.reason_code, self.reason_code)

    @property
    def reasons(self) -> List[str]:
        return [self.reason] + list(self.notes)

    flagging_reasons = reasons

    def vrs_table(self) -> str:
        """Formatted validity report."""
        def fmt(ix, signed=False):
            if ix is None or ix.flag == "not_estimable" or np.isnan(ix.value):
                return f"not estimable ({ix.note})" if ix is not None and ix.note else "not estimable"
            f = "{:+.3f}" if signed else "{:.3f}"
            return f"{f.format(ix.value)} [{f.format(ix.ci_lower)}, {f.format(ix.ci_upper)}]"

        rows = [
            ("Model", self.model_name or "[not specified]"),
            ("Benchmark", self.benchmark_name or "[not specified]"),
            ("Items / excluded / scored", f"{self.n_items} / {self.n_excluded} / {self.n_scored}"
             + (f"  ({self.n_missing} missing values)" if self.n_missing else "")),
            ("Correct / wrong", f"{self.n_correct} / {self.n_incorrect}"),
            ("Accuracy", f"{self.accuracy:.3f}" if self.n_scored else "n/a"),
            ("Confidence elicitation", self.elicitation_method or "[not specified]"),
            ("Confidence format", self.confidence_format or "[not specified]"),
            ("Cut point", self.binarisation_threshold),
            ("Probe timing", self.probe_timing or "[not specified]"),
            ("2x2 table", f"a={self.a} (correct, kept)  b={self.b} (wrong, kept)  "
                          f"c={self.c} (correct, withdrawn)  d={self.d} (wrong, withdrawn)"),
            ("Kept-error rate L", fmt(self.L)),
            ("Withdrawn-correct rate Fp", fmt(self.Fp)),
            ("Item sensitivity J", fmt(self.J, True)),
            ("One-sided mid-p, J > 0", "n/a" if np.isnan(self.p_signal) else f"{self.p_signal:.4f}"),
            ("Binary AUROC (1 + J) / 2", "n/a" if np.isnan(self.auroc_binary) else f"{self.auroc_binary:.3f}"),
        ]
        r = self.r_conf_correct
        if r is not None and not np.isnan(r.value):
            rows.append(("r(confidence, correct)", f"{r.value:+.3f} [{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]"))
        rows.append(("Tier", self.tier))
        rows.append(("Reason", self.reason))
        rows.append(("Flags", ", ".join(self.flags) if self.flags else "none"))
        if self.notes:
            rows.append(("Notes", "; ".join(self.notes)))
        w = max(len(k) for k, _ in rows)
        bar = "=" * (w + 60)
        out = [bar, "VALIDITY REPORT: confidence screen (validity-screen 0.4)", bar]
        out += [f"  {k:<{w}}  {v}" for k, v in rows]
        out.append(bar)
        return "\n".join(out)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in [
            "model_name", "benchmark_name", "binarisation_threshold", "n_items", "n_excluded",
            "n_missing", "n_scored", "n_correct", "n_incorrect", "accuracy", "a", "b", "c", "d",
            "p_signal", "p_inversion", "auroc_binary", "tier", "reason_code"]}
        d["reason"] = self.reason
        d["flags"] = list(self.flags)
        d["notes"] = list(self.notes)
        for nm in ["trin", "L", "Fp", "J", "r_conf_correct"]:
            ix = getattr(self, nm)
            if ix is not None:
                d[f"{nm}_value"], d[f"{nm}_ci_lower"], d[f"{nm}_ci_upper"], d[f"{nm}_flag"] = (
                    ix.value, ix.ci_lower, ix.ci_upper, ix.flag)
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


# ============================================================
# Input checks
# ============================================================

def _as_binary(x, name: str, scored: np.ndarray) -> np.ndarray:
    """Return a bool array; raise if any scored value is missing or not 0/1."""
    arr = np.asarray(x)
    if arr.dtype == bool:
        return arr
    try:
        f = arr.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be 0/1 or True/False; got non-numeric values. "
                         f"Map labels to 0/1 first.") from None
    bad_nan = np.isnan(f) & scored
    if bad_nan.any():
        raise ValueError(f"{name} has {int(bad_nan.sum())} missing values on rows that are scored. "
                         f"Mark those rows as not answered (answered=0); missing values are never "
                         f"guessed.")
    vals = np.unique(f[scored & ~np.isnan(f)])
    if not np.all(np.isin(vals, [0.0, 1.0])):
        hint = (" It looks like continuous confidence: pass cut=<threshold> to screen(), "
                "chosen in advance from the threshold you intend to use.") if name == "confidence" else ""
        raise ValueError(f"{name} must be binary (0/1); found values such as "
                         f"{', '.join(f'{v:g}' for v in vals[:5])}.{hint}")
    return np.nan_to_num(f, nan=0.0).astype(bool)


# ============================================================
# Screen
# ============================================================

def screen(
    correct,
    confidence,
    answered=None,
    cut: Optional[float] = None,
    model_name: str = "",
    benchmark_name: str = "",
    elicitation_method: str = "",
    confidence_format: str = "",
    binarisation_threshold: Optional[str] = None,
    probe_timing: str = "",
    alpha: float = 0.05,
    min_j: float = 0.0,
) -> ScreenResult:
    """Run the validity screen.

    Parameters
    ----------
    correct : array-like of 0/1 or bool
        Item correctness.
    confidence : array-like
        Binary (1 = kept / high confidence, 0 = withdrawn / low), or continuous if
        `cut` is given. Missing values are allowed only on rows marked answered=0,
        or, with `cut`, are excluded and counted.
    answered : array-like of 0/1 or bool, optional
        0 marks rows to exclude before scoring: declines, unparseable outputs, and
        values the pipeline supplied rather than the model (e.g. a parser default).
    cut : float, optional
        Cut point for continuous confidence: confidence >= cut counts as kept. Choose
        it before looking at results, from the threshold you intend to use.
    alpha : float
        Two-sided level for the reported intervals (default .05). The certification
        guard (.90), the flags (95% intervals) and the one-sided signal test (.025) are fixed.
    min_j : float
        Optional smallest item sensitivity of interest (default 0). If > 0, Valid also
        requires the lower 95% Newcombe bound of J to exceed min_j; otherwise the verdict is
        Indeterminate with reason "sensitivity_below_minimum". Choose it before seeing data.
    """
    if not (0 <= min_j < 1):
        raise ValueError("min_j must be in [0, 1)")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    corr_arr = np.asarray(correct)
    conf_arr = np.asarray(confidence)
    for nm, arr in [("correct", corr_arr), ("confidence", conf_arr)]:
        if arr.ndim != 1:
            raise ValueError(f"{nm} must be one-dimensional (one value per item)")
    n_items = len(corr_arr)
    if len(conf_arr) != n_items:
        raise ValueError("correct and confidence must have equal length")
    if answered is None:
        ans = np.ones(n_items, dtype=bool)
    else:
        a_arr = np.asarray(answered)
        if a_arr.ndim != 1 or len(a_arr) != n_items:
            raise ValueError("answered must be one-dimensional with one value per item")
        ans = _as_binary(a_arr, "answered", np.ones(n_items, dtype=bool))

    n_missing = 0
    if cut is not None:
        try:
            cut = float(cut)
        except (TypeError, ValueError):
            raise ValueError("cut must be a number") from None
        if not np.isfinite(cut):
            raise ValueError("cut must be a finite number")
        try:
            cf = conf_arr.astype(float)
        except (TypeError, ValueError):
            raise ValueError("with cut=..., confidence must be numeric") from None
        known = ~np.isnan(cf)
        if cut > 1 and np.all(np.isin(cf[known], [0.0, 1.0])):
            raise ValueError(f"confidence looks binary (only 0 and 1) but cut={cut:g}; "
                             f"omit cut for binary confidence.")
        if known.any():
            cmax = float(np.nanmax(cf))
            if cut > 1 and cmax <= 1:
                raise ValueError(f"confidence looks like a 0-1 scale (maximum {cmax:g}) but cut={cut:g}; "
                                 f"use a cut on the same scale (e.g. 0.9) or rescale the confidence.")
            if cut <= 1 and cmax > 1:
                raise ValueError(f"confidence looks like a 0-100 scale (maximum {cmax:g}) but cut={cut:g}; "
                                 f"use a cut on the same scale (e.g. 90) or rescale the confidence.")
        n_missing = int((ans & ~known).sum())
        ans = ans & known
        conf_b = np.where(known, cf >= cut, False)
        if binarisation_threshold is None:
            binarisation_threshold = f">= {cut:g}"
    else:
        conf_b = _as_binary(conf_arr, "confidence", ans)
        if binarisation_threshold is None:
            binarisation_threshold = "N/A (binary)"
    corr_b = _as_binary(corr_arr, "correct", ans)

    res = ScreenResult(model_name=model_name, benchmark_name=benchmark_name,
                       elicitation_method=elicitation_method, confidence_format=confidence_format,
                       binarisation_threshold=binarisation_threshold, probe_timing=probe_timing,
                       n_items=n_items, n_excluded=int((~ans).sum()), n_missing=n_missing)
    cor, conf = corr_b[ans], conf_b[ans]
    n = len(cor)
    nc = int(cor.sum())
    ni = n - nc
    a, b = int((cor & conf).sum()), int((~cor & conf).sum())
    c, d = int((cor & ~conf).sum()), int((~cor & ~conf).sum())
    res.n_scored, res.n_correct, res.n_incorrect = n, nc, ni
    res.accuracy = nc / n if n else float("nan")
    res.a, res.b, res.c, res.d = a, b, c, d

    trin = max(a + b, c + d) / n if n else float("nan")
    res.trin = IndexResult("TRIN", trin, trin, trin, n, "warning" if n and trin >= TRIN_CUT else "ok")
    if n and trin >= TRIN_CUT:
        res.flags.append("fixed_responding")

    def rate(k, m, name, note):
        ix = IndexResult(name, n=m)
        if m >= MIN_N:
            lo, hi = wilson_ci(k, m, alpha)
            ix.value, ix.ci_lower, ix.ci_upper = k / m, lo, hi
            if ix.value >= RESPONSE_SET_CUT:
                guard_lo = wilson_ci(k, m, 0.05)[0]
                ix.flag = "certified" if guard_lo > GUARD else "suspected"
        else:
            ix.flag, ix.note = "not_estimable", note
        return ix

    res.L = rate(b, ni, "L", f"{ni} errors < {MIN_N}")
    res.Fp = rate(c, nc, "Fp", f"{nc} correct < {MIN_N}")

    res.J = IndexResult("J", n=n)
    if nc >= MIN_N and ni >= MIN_N:
        j = (a / nc) - (b / ni)
        lo, hi = newcombe_diff_ci(a, nc, b, ni, alpha)   # P(keep|correct) - P(keep|wrong)
        res.J.value, res.J.ci_lower, res.J.ci_upper = j, lo, hi
        res.auroc_binary = (1 + j) / 2
        res.p_signal, res.p_inversion = midp_signal(a, b, c, d)
    else:
        res.J.flag, res.J.note = "not_estimable", f"needs >= {MIN_N} correct and >= {MIN_N} errors"

    res.r_conf_correct = IndexResult("r", n=n)
    if n > 3 and conf.std() > 0 and cor.std() > 0:
        rv, pv = sp_stats.pearsonr(conf.astype(float), cor.astype(float))
        zr = np.arctanh(np.clip(rv, -0.999999, 0.999999))
        zc = sp_stats.norm.ppf(1 - alpha / 2)
        se = 1 / np.sqrt(n - 3)
        res.r_conf_correct.value = float(rv)
        res.r_conf_correct.ci_lower = float(np.tanh(zr - zc * se))
        res.r_conf_correct.ci_upper = float(np.tanh(zr + zc * se))
        res.r_conf_correct.note = f"p = {pv:.4f}"
    else:
        res.r_conf_correct.flag, res.r_conf_correct.note = "not_estimable", "no variance"

    # ---- tier ----
    if n == 0:
        res.tier, res.reason_code = "Uncertifiable", "no_scored_rows"
        return res
    if res.L.flag == "certified":
        res.tier, res.reason_code = "Invalid", "blanket_confidence"
    elif res.Fp.flag == "certified":
        res.tier, res.reason_code = "Invalid", "blanket_withdrawal"
    elif res.J.flag == "not_estimable":
        res.tier = "Uncertifiable"
        res.reason_code = "too_few_errors" if ni < MIN_N else "too_few_correct"
    elif res.p_inversion < SIGNAL_ALPHA:
        res.tier, res.reason_code = "Invalid", "inversion"
    elif res.L.flag == "suspected":
        res.tier, res.reason_code = "Indeterminate", "blanket_confidence_suspected"
    elif res.Fp.flag == "suspected":
        res.tier, res.reason_code = "Indeterminate", "blanket_withdrawal_suspected"
    elif res.p_signal < SIGNAL_ALPHA:
        if min_j > 0 and not (newcombe_diff_ci(a, nc, b, ni, 0.05)[0] > min_j):
            res.tier, res.reason_code = "Indeterminate", "sensitivity_below_minimum"
        else:
            res.tier, res.reason_code = "Valid", "signal_detected"
    elif res.J.value < 0:
        res.tier, res.reason_code = "Indeterminate", "inversion_suspected"
    else:
        res.tier, res.reason_code = "Indeterminate", "no_detectable_sensitivity"

    # ---- flags ----
    if nc >= MIN_N and res.Fp.value >= LOW_COVERAGE_CUT:
        res.flags.append("low_coverage")
    if res.tier == "Valid":
        hiL = wilson_ci(b, ni, 0.05)[1] if ni >= MIN_N else float("nan")
        hiF = wilson_ci(c, nc, 0.05)[1] if nc >= MIN_N else float("nan")
        if hiL >= RESPONSE_SET_CUT or hiF >= RESPONSE_SET_CUT:
            res.flags.append("response_set_not_excluded")
    if res.n_missing:
        res.notes.append(f"{res.n_missing} rows excluded for missing values")
    return res


# ============================================================
# Batch, summary, binarisation
# ============================================================

def screen_batch(models: dict, benchmark_name: str = "", **kwargs) -> List[ScreenResult]:
    """Screen several models. `models` maps name -> dict with 'correct', 'confidence'
    and optionally 'answered'."""
    return [screen(d["correct"], d["confidence"], answered=d.get("answered"), model_name=name,
                   benchmark_name=benchmark_name, **kwargs) for name, d in models.items()]


def summary_table(results: List[ScreenResult]) -> str:
    """One line per model."""
    def v(ix, signed=False):
        if ix is None or np.isnan(ix.value):
            return "--"
        return f"{ix.value:+.3f}" if signed else f"{ix.value:.3f}"
    header = (f"{'Model':<28} {'Tier':<14} {'Err':>5} {'L':>6} {'Fp':>6} {'J':>7} "
              f"{'p(J>0)':>8}  {'Reason / flags'}")
    sep = "-" * (len(header) + 30)
    lines = [sep, header, sep]
    order = {"Invalid": 0, "Indeterminate": 1, "Uncertifiable": 2, "Valid": 3}
    for r in sorted(results, key=lambda x: order.get(x.tier, 9)):
        p = "--" if np.isnan(r.p_signal) else f"{r.p_signal:.4f}"
        extra = r.reason_code + (" | " + ", ".join(r.flags) if r.flags else "")
        lines.append(f"{r.model_name[:28]:<28} {r.tier:<14} {r.n_incorrect:>5} {v(r.L):>6} {v(r.Fp):>6} "
                     f"{v(r.J, True):>7} {p:>8}  {extra}")
    lines.append(sep)
    return "\n".join(lines)


def binarise(confidence, threshold: float) -> np.ndarray:
    """Binarise continuous confidence: True where confidence >= threshold.

    The threshold has no default: choose it in advance, from the threshold you intend
    to use. Missing values raise an error; to exclude them, pass the raw confidence to
    screen(..., cut=threshold) instead, which excludes and counts them.
    """
    conf = np.asarray(confidence, dtype=float)
    if np.isnan(conf).any():
        raise ValueError(f"confidence has {int(np.isnan(conf).sum())} missing values; use "
                         f"screen(correct, confidence, cut={threshold:g}) to exclude and count them.")
    return conf >= threshold


# ============================================================
# Planning: certification counts and power
# ============================================================

def errors_needed(withdrawn: int = 0, guard: float = GUARD) -> int:
    """Smallest number of errors at which blanket confidence can be certified when
    `withdrawn` of those errors were withdrawn (kept = errors - withdrawn): the kept
    share must be >= .95 and its Wilson lower bound > .90.

    0 -> 35, 1 -> 53, 2 -> 69, 3 -> 84, 4 -> 99, 5 -> 113, 6 -> 127, 7 -> 140.
    This is the count at which an observed table can certify. For planning, use
    power_invalid() or items_needed(), which account for sampling.
    """
    if isinstance(withdrawn, float) or not isinstance(withdrawn, (int, np.integer)) or withdrawn < 0:
        raise TypeError("errors_needed() takes a whole number of withdrawn errors (0.4 changed this; "
                        "for a true kept-error rate use power_invalid() or items_needed())")
    n = withdrawn + 1
    while True:
        kept = n - withdrawn
        if kept / n >= RESPONSE_SET_CUT and wilson_ci(kept, n)[0] > guard:
            return n
        n += 1


def _check_prob(x, name, open_low=False, open_high=False):
    if not isinstance(x, (int, float, np.integer, np.floating)) or not np.isfinite(x):
        raise ValueError(f"{name} must be a number")
    lo_ok = x > 0 if open_low else x >= 0
    hi_ok = x < 1 if open_high else x <= 1
    if not (lo_ok and hi_ok):
        raise ValueError(f"{name} must be between 0 and 1")


def _check_count(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0:
        raise ValueError(f"{name} must be a whole number >= 0")


def power_invalid(true_L: float, n_errors: int) -> float:
    """Exact probability that blanket confidence is certified (L >= .95 and Wilson lower
    bound > .90) when the true kept-error rate is `true_L` and there are `n_errors` errors."""
    _check_prob(true_L, "true_L")
    _check_count(n_errors, "n_errors")
    k_min = _kept_needed(n_errors)
    if k_min is None:
        return 0.0
    return float(sp_stats.binom.sf(k_min - 1, n_errors, true_L))


def _kept_needed(n_errors: int) -> Optional[int]:
    """Smallest number of kept errors that certifies blanket confidence out of n_errors
    (both conditions are monotone in the kept count)."""
    if n_errors < MIN_N:
        return None
    for kk in range(int(np.ceil(RESPONSE_SET_CUT * n_errors - 1e-9)), n_errors + 1):
        if kk / n_errors >= RESPONSE_SET_CUT and wilson_ci(kk, n_errors)[0] > GUARD:
            return kk
    return None


def power_valid(n_items: int, accuracy: float, keep_correct: float, keep_wrong: float,
                tol: float = 1e-10) -> float:
    """Exact probability of a Valid verdict (errors fixed at round(n_items * (1 - accuracy))),
    for a model that keeps correct answers with probability `keep_correct` and errors with
    probability `keep_wrong`. With keep_correct == keep_wrong this is the false-Valid rate."""
    _check_count(n_items, "n_items")
    _check_prob(accuracy, "accuracy")
    _check_prob(keep_correct, "keep_correct")
    _check_prob(keep_wrong, "keep_wrong")
    ni = int(round(n_items * (1 - accuracy)))
    nc = n_items - ni
    if ni < MIN_N or nc < MIN_N:
        return 0.0
    pa = sp_stats.binom.pmf(np.arange(nc + 1), nc, keep_correct)
    pb = sp_stats.binom.pmf(np.arange(ni + 1), ni, keep_wrong)
    A = np.nonzero(pa > tol)[0]
    total = 0.0
    for b in np.nonzero(pb > tol)[0]:
        a = A
        c, d = nc - a, ni - b
        L, Fp = b / ni, c / nc
        Lcert = (L >= RESPONSE_SET_CUT)
        Fcert = (Fp >= RESPONSE_SET_CUT)
        K = a + b
        N = nc + ni
        pmf = sp_stats.hypergeom.pmf(a, N, nc, K)
        p_sig = sp_stats.hypergeom.sf(a, N, nc, K) + 0.5 * pmf
        valid = (~Lcert) & (~Fcert) & (p_sig < SIGNAL_ALPHA)
        total += float((pa[a] * pb[b] * valid).sum())
    return total


def items_needed(accuracy: float, true_L: float = 1.0, power: float = 0.90,
                 max_errors: int = 5000) -> Optional[int]:
    """Items needed for a `power` chance of certifying blanket confidence when the true
    kept-error rate is `true_L` and accuracy is `accuracy`. Returns None if the target
    cannot be reached (e.g. true_L at or near .95, where certification succeeds at most
    about half the time at any sample size). Power is not monotone in the number of
    errors; the value returned is the smallest count from which power stays >= target
    over the next 50 counts."""
    _check_prob(accuracy, "accuracy")
    _check_prob(true_L, "true_L")
    _check_prob(power, "power", open_low=True, open_high=True)
    if accuracy >= 1:
        return None
    run = 0
    first = None
    for e in range(MIN_N, max_errors):
        if power_invalid(true_L, e) >= power:
            first = e if run == 0 else first
            run += 1
            if run >= 50:
                return int(np.ceil(round(first / (1 - accuracy), 9)))
        else:
            run = 0
    return None
