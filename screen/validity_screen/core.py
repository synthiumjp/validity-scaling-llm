"""
validity_screen.core
====================

Stage A validity screen for binary (or binarised) LLM confidence signals,
as specified in Cacioli (2026), "Screen Before You Interpret", arXiv:2604.17714v2.

Tier rules (version 0.3.0)
--------------------------
Each index is computed only when its own denominator reaches MIN_N (10):
correct items for Fp, errors for L, both for RBS.

  Fp >= .50 : Invalid if Wilson lower bound > .40, else Indeterminate
  L  >= .95 : Invalid if Wilson lower bound > .90, else Indeterminate
  RBS > 0   : Invalid if Newcombe interval excludes 0, else Indeterminate
  TRIN >= .95 : structural warning only (never decides a tier)

Tier = Invalid if any index is Invalid; else Indeterminate if any is;
else Uncertifiable if RBS cannot be estimated; else Valid only if the RBS
interval lies entirely below 0 (Youden's J significantly > 0), otherwise
Indeterminate ("no detectable item sensitivity").

`require_signal=False` reproduces the version 0.2 rule (Valid = no flag);
the v0.2 tier is always reported as `tier_v02`.
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

MIN_N = 10
FP_CUT, FP_GUARD = 0.50, 0.40
L_CUT, L_GUARD = 0.95, 0.90
TRIN_CUT = 0.95


# ============================================================
# Intervals
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


# ============================================================
# Result containers (field names kept compatible with 0.1.0)
# ============================================================

@dataclass
class IndexResult:
    """Result for a single index. `flag` is one of
    'ok', 'invalid', 'indeterminate', 'not_estimable', 'warning'."""
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
    n_excluded: int = 0
    n_scored: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    accuracy: float = float("nan")
    a: int = 0   # correct, high confidence
    b: int = 0   # incorrect, high confidence
    c: int = 0   # correct, low confidence
    d: int = 0   # incorrect, low confidence
    trin: Optional[IndexResult] = None
    L: Optional[IndexResult] = None
    Fp: Optional[IndexResult] = None
    rbs: Optional[IndexResult] = None
    r_conf_correct: Optional[IndexResult] = None
    youden_j: float = float("nan")
    auroc_binary: float = float("nan")
    tier: str = ""
    tier_v02: str = ""
    flagging_reasons: List[str] = field(default_factory=list)

    @property
    def r(self) -> Optional[IndexResult]:
        return self.r_conf_correct

    @property
    def reasons(self) -> List[str]:
        return self.flagging_reasons

    def vrs_table(self) -> str:
        """Formatted VRS Table (Validity Report for confidence Screening)."""
        def fmt(ix, signed=False):
            if ix is None or ix.flag == "not_estimable" or np.isnan(ix.value):
                return f"not estimable ({ix.note})" if ix is not None and ix.note else "not estimable"
            f = "{:+.3f}" if signed else "{:.3f}"
            return f"{f.format(ix.value)} [{f.format(ix.ci_lower)}, {f.format(ix.ci_upper)}]"

        rows = [
            ("Model", self.model_name or "[not specified]"),
            ("Benchmark", self.benchmark_name or "[not specified]"),
            ("Items / excluded / scored", f"{self.n_items} / {self.n_excluded} / {self.n_scored}"),
            ("Correct / incorrect", f"{self.n_correct} / {self.n_incorrect}"),
            ("Accuracy", f"{self.accuracy:.3f}" if self.n_scored else "n/a"),
            ("Confidence elicitation", self.elicitation_method or "[not specified]"),
            ("Confidence format", self.confidence_format or "[not specified]"),
            ("Cut point", self.binarisation_threshold),
            ("Probe timing", self.probe_timing or "[not specified]"),
            ("2x2 table", f"a={self.a}, b={self.b}, c={self.c}, d={self.d}"),
        ]
        if self.trin is not None and not np.isnan(self.trin.value):
            direction = "fixed-high" if self.a + self.b >= self.c + self.d else "fixed-low"
            warn = " (structural warning)" if self.trin.flag == "warning" else ""
            rows.append(("TRIN", f"{self.trin.value:.3f} {direction}{warn}"))
        rows += [("L", fmt(self.L)), ("Fp", fmt(self.Fp)), ("RBS", fmt(self.rbs, True))]
        r = self.r_conf_correct
        if r is not None and not np.isnan(r.value):
            rows.append(("r(confidence, correct)", f"{r.value:+.3f} [{r.ci_lower:+.3f}, {r.ci_upper:+.3f}], {r.note}"))
        rows.append(("Tier", self.tier + ("" if self.tier == self.tier_v02 else f"  (v0.2 rule: {self.tier_v02})")))
        rows.append(("Reasons", "; ".join(self.flagging_reasons) if self.flagging_reasons else "none"))
        w = max(len(k) for k, _ in rows)
        bar = "=" * (w + 52)
        out = [bar, "VRS TABLE: Validity Report for confidence Screening", bar]
        out += [f"  {k:<{w}}  {v}" for k, v in rows]
        out.append(bar)
        return "\n".join(out)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in [
            "model_name", "benchmark_name", "binarisation_threshold", "n_items", "n_excluded", "n_scored",
            "n_correct", "n_incorrect", "accuracy", "a", "b", "c", "d", "youden_j", "auroc_binary",
            "tier", "tier_v02"]}
        d["flagging_reasons"] = list(self.flagging_reasons)
        for nm in ["trin", "L", "Fp", "rbs", "r_conf_correct"]:
            ix = getattr(self, nm)
            if ix is not None:
                d[f"{nm}_value"], d[f"{nm}_ci_lower"], d[f"{nm}_ci_upper"], d[f"{nm}_flag"] = (
                    ix.value, ix.ci_lower, ix.ci_upper, ix.flag)
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


# ============================================================
# Screen
# ============================================================

def screen(
    correct,
    confidence,
    answered=None,
    model_name: str = "",
    benchmark_name: str = "",
    elicitation_method: str = "",
    confidence_format: str = "",
    binarisation_threshold: str = "N/A",
    probe_timing: str = "",
    alpha: float = 0.05,
    require_signal: bool = True,
) -> ScreenResult:
    """Run the Stage A screen.

    Parameters
    ----------
    correct : array-like of 0/1 or bool
        Item correctness.
    confidence : array-like of 0/1 or bool
        1 = high confidence (KEEP), 0 = low (WITHDRAW). Binarise continuous
        confidence first with `binarise(conf, cut)` at a cut point chosen in advance.
    answered : array-like of 0/1 or bool, optional
        0 marks rows to exclude before scoring: declines, unparseable outputs, and
        confidence values supplied by the pipeline (e.g. a parser default).
    require_signal : bool
        True (default, v0.3): Valid requires the RBS interval to lie below 0.
        False: version 0.2 rule (Valid = no flag fired).
    """
    correct = np.asarray(correct).astype(bool)
    confidence = np.asarray(confidence).astype(bool)
    answered = np.ones_like(correct, dtype=bool) if answered is None else np.asarray(answered).astype(bool)
    if not (len(correct) == len(confidence) == len(answered)):
        raise ValueError("correct, confidence and answered must have equal length")

    res = ScreenResult(model_name=model_name, benchmark_name=benchmark_name,
                       elicitation_method=elicitation_method, confidence_format=confidence_format,
                       binarisation_threshold=binarisation_threshold, probe_timing=probe_timing,
                       n_items=len(correct), n_excluded=int((~answered).sum()))
    cor, conf = correct[answered], confidence[answered]
    n = len(cor)
    nc = int(cor.sum())
    ni = n - nc
    a, b = int((cor & conf).sum()), int((~cor & conf).sum())
    c, d = int((cor & ~conf).sum()), int((~cor & ~conf).sum())
    res.n_scored, res.n_correct, res.n_incorrect = n, nc, ni
    res.accuracy = nc / n if n else float("nan")
    res.a, res.b, res.c, res.d = a, b, c, d
    reasons = res.flagging_reasons

    trin = max(a + b, c + d) / n if n else float("nan")
    res.trin = IndexResult("TRIN", trin, trin, trin, n, "warning" if n and trin >= TRIN_CUT else "ok")

    res.Fp = IndexResult("Fp", n=nc)
    if nc >= MIN_N:
        fp = c / nc
        lo, hi = wilson_ci(c, nc, alpha)
        res.Fp.value, res.Fp.ci_lower, res.Fp.ci_upper = fp, lo, hi
        if fp >= FP_CUT:
            res.Fp.flag = "invalid" if lo > FP_GUARD else "indeterminate"
            reasons.append(f"Fp = {fp:.3f} (Wilson lower {lo:.3f})")
    else:
        res.Fp.flag, res.Fp.note = "not_estimable", f"{nc} correct items < {MIN_N}"
        reasons.append(f"Fp not estimable: {res.Fp.note}")

    res.L = IndexResult("L", n=ni)
    if ni >= MIN_N:
        lv = b / ni
        lo, hi = wilson_ci(b, ni, alpha)
        res.L.value, res.L.ci_lower, res.L.ci_upper = lv, lo, hi
        if lv >= L_CUT:
            res.L.flag = "invalid" if lo > L_GUARD else "indeterminate"
            reasons.append(f"L = {lv:.3f} (Wilson lower {lo:.3f}; "
                           f"{'certified' if lo > L_GUARD else 'too few errors to certify'})")
    else:
        res.L.flag, res.L.note = "not_estimable", f"{ni} errors < {MIN_N}"
        reasons.append(f"L not estimable: {res.L.note}")

    res.rbs = IndexResult("RBS", n=n)
    if nc >= MIN_N and ni >= MIN_N:
        rbs = c / nc - d / ni
        lo, hi = newcombe_diff_ci(c, nc, d, ni, alpha)
        res.rbs.value, res.rbs.ci_lower, res.rbs.ci_upper = rbs, lo, hi
        res.youden_j = -rbs
        res.auroc_binary = (1 - rbs) / 2
        if rbs > 0:
            res.rbs.flag = "invalid" if lo > 0 else "indeterminate"
            reasons.append(f"RBS = {rbs:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    else:
        res.rbs.flag, res.rbs.note = "not_estimable", "needs >= 10 correct and >= 10 errors"

    res.r_conf_correct = IndexResult("r", n=n)
    if n > 3 and conf.std() > 0 and cor.std() > 0:
        rv, pv = sp_stats.pearsonr(conf.astype(float), cor.astype(float))
        zr = np.arctanh(np.clip(rv, -0.999999, 0.999999))
        zc = sp_stats.norm.ppf(1 - alpha / 2)
        se = 1 / np.sqrt(n - 3)
        res.r_conf_correct.value = float(rv)
        res.r_conf_correct.ci_lower, res.r_conf_correct.ci_upper = float(np.tanh(zr - zc * se)), float(np.tanh(zr + zc * se))
        res.r_conf_correct.note = f"p = {pv:.4f}"
    else:
        res.r_conf_correct.flag, res.r_conf_correct.note = "not_estimable", "no variance"

    flags = [res.Fp.flag, res.L.flag, res.rbs.flag]
    if "invalid" in flags:
        res.tier = "Invalid"
    elif "indeterminate" in flags:
        res.tier = "Indeterminate"
    elif res.rbs.flag == "not_estimable":
        res.tier = "Uncertifiable"
    else:
        res.tier = "Valid"
    res.tier_v02 = res.tier
    if require_signal and res.tier == "Valid" and not (res.rbs.ci_upper < 0):
        res.tier = "Indeterminate"
        reasons.append(f"no detectable item sensitivity: RBS = {res.rbs.value:+.3f} "
                       f"[{res.rbs.ci_lower:+.3f}, {res.rbs.ci_upper:+.3f}] includes 0")
    return res


# ============================================================
# Batch, summary, binarisation, planning
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
    header = f"{'Model':<28} {'Tier':<14} {'Err':>5} {'L':>6} {'Fp':>6} {'RBS':>7} {'r':>7} {'TRIN':>6}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    order = {"Invalid": 0, "Indeterminate": 1, "Uncertifiable": 2, "Valid": 3}
    for r in sorted(results, key=lambda x: order.get(x.tier, 9)):
        lines.append(f"{r.model_name[:28]:<28} {r.tier:<14} {r.n_incorrect:>5} {v(r.L):>6} {v(r.Fp):>6} "
                     f"{v(r.rbs, True):>7} {v(r.r_conf_correct, True):>7} {v(r.trin):>6}")
    lines.append(sep)
    return "\n".join(lines)


def binarise(confidence, threshold: float = 0.5, method: str = "fixed") -> np.ndarray:
    """Binarise continuous confidence: True where confidence >= threshold.

    Choose the cut point in advance (ideally the deployment threshold) and report it.
    method='median' is deprecated: with tied confidence values its verdict depends
    on tie handling, and an exact 50/50 split puts Fp near .50 by construction.
    """
    confidence = np.asarray(confidence, dtype=float)
    if method == "median":
        warnings.warn("median binarisation is deprecated in validity-screen 0.3: its verdict depends on "
                      "tie handling. Use a fixed cut point chosen in advance.", DeprecationWarning, stacklevel=2)
        threshold = float(np.median(confidence))
    elif method != "fixed":
        raise ValueError("method must be 'fixed' (or the deprecated 'median')")
    return confidence >= threshold


def _wilson_lb_at(p: float, n: float, z: float = 1.959963984540054) -> float:
    den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def errors_needed(L: float = 1.0, guard: float = L_GUARD, max_n: int = 100000) -> Optional[int]:
    """Smallest number of errors at which an observed L has a Wilson lower bound above
    `guard` (default .90), i.e. can be certified as blanket confidence. Returns None if
    L <= guard. Examples: L=1.00 -> 35, .97 -> 71, .96 -> 97, .95 -> 139."""
    if L <= guard:
        return None
    for n in range(1, max_n):
        if _wilson_lb_at(L, n) > guard:
            return n
    return None


def items_needed(accuracy: float, L: float = 1.0, guard: float = L_GUARD) -> Optional[int]:
    """Items needed to expect enough errors to certify blanket confidence at a given accuracy."""
    e = errors_needed(L, guard)
    if e is None or accuracy >= 1:
        return None
    return int(np.ceil(e / (1 - accuracy)))
