"""
validity_screen v0.3.0
======================

Stage A validity screen for binary (or binarised) LLM confidence signals.

Changes from v0.1.0 (all deliberate, documented in CHANGES section below):
  1. Non-answers are excluded before the 2x2 table is built and are reported
     separately as an abstention count. A declined or unparseable item carries
     no confidence judgement about an answer, so it cannot be a "confident error".
  2. The v0.1 guard "any cell < 5 -> Insufficient data" is replaced by
     per-index denominator guards. The old guard blocked exactly the response
     sets the screen exists to detect: blanket confidence empties cell d by
     definition. Each index is now computed when its own denominator is >= MIN_N.
  3. The RBS interval is the Newcombe hybrid score interval for a difference of
     two independent proportions (v0.1 used a Wald interval, which collapses to
     zero width when L = 1 or Fp = 0).
  4. The output reports the standard signal-detection equivalents explicitly:
     Youden's J = -RBS, binary AUROC = (1 - RBS) / 2, and r = phi.
  5. No tier can be overridden. The tier is a pure function of the data and the
     constants below.

Change in v0.3.0:
  6. Valid requires positive evidence of item sensitivity: no flag fires AND the
     RBS interval lies entirely below 0 (Youden's J significantly > 0). Without
     that evidence the tier is Indeterminate ("no detectable item sensitivity").
     Under v0.2 an uninformative random policy was classified Valid in 30-50% of
     simulated draws. The v0.2 tier is still returned as `tier_v02`.

Thresholds are unchanged from the published protocol (arXiv:2604.17714v1):
  Fp >= .50 with Wilson lower bound > .40  -> Invalid
  Fp >= .50 otherwise                     -> Indeterminate
  L  >= .95 with Wilson lower bound > .90  -> Invalid
  L  >= .95 otherwise                     -> Indeterminate
  RBS > 0 with interval excluding 0       -> Invalid
  RBS > 0 otherwise                       -> Indeterminate
  TRIN >= .95                             -> structural warning only
"""

__version__ = "0.3.0"

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

MIN_N = 10            # minimum denominator for L (errors) or Fp (correct items)
FP_CUT, FP_GUARD = 0.50, 0.40
L_CUT, L_GUARD = 0.95, 0.90
TRIN_CUT = 0.95


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int,
                     alpha: float = 0.05) -> Tuple[float, float]:
    """Newcombe (1998) hybrid score interval for p1 - p2, independent samples."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, alpha)
    l2, u2 = wilson_ci(k2, n2, alpha)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (lo, hi)


@dataclass
class Index:
    name: str
    value: float = np.nan
    lo: float = np.nan
    hi: float = np.nan
    n: int = 0
    flag: str = "ok"   # ok | invalid | indeterminate | not_estimable | warning


@dataclass
class ScreenResult:
    model_name: str = ""
    n_items: int = 0
    n_abstained: int = 0
    n_scored: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    accuracy: float = np.nan
    a: int = 0   # correct, high confidence
    b: int = 0   # incorrect, high confidence
    c: int = 0   # correct, low confidence
    d: int = 0   # incorrect, low confidence
    trin: Index = None
    L: Index = None
    Fp: Index = None
    rbs: Index = None
    r: Index = None
    youden_j: float = np.nan
    auroc_binary: float = np.nan
    tier: str = ""
    tier_v02: str = ""
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {k: getattr(self, k) for k in
               ["model_name", "n_items", "n_abstained", "n_scored", "n_correct",
                "n_incorrect", "accuracy", "a", "b", "c", "d", "youden_j",
                "auroc_binary", "tier", "tier_v02"]}
        for nm in ["trin", "L", "Fp", "rbs", "r"]:
            ix = getattr(self, nm)
            out[f"{nm}"], out[f"{nm}_lo"], out[f"{nm}_hi"], out[f"{nm}_flag"] = (
                ix.value, ix.lo, ix.hi, ix.flag)
        out["reasons"] = "; ".join(self.reasons)
        return out


def screen(correct, confidence, answered=None, model_name: str = "",
           alpha: float = 0.05, require_signal: bool = True) -> ScreenResult:
    """Run the Stage A screen.

    correct    : array of 0/1, item correctness
    confidence : array of 0/1, 1 = high confidence (KEEP), 0 = low (WITHDRAW)
    answered   : optional array of 0/1; rows with 0 are abstentions or
                 unparseable outputs and are excluded from the 2x2 table
    """
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=bool)
    if answered is None:
        answered = np.ones_like(correct, dtype=bool)
    answered = np.asarray(answered, dtype=bool)
    if not (len(correct) == len(confidence) == len(answered)):
        raise ValueError("correct, confidence and answered must have equal length")

    res = ScreenResult(model_name=model_name, n_items=len(correct),
                       n_abstained=int((~answered).sum()))
    cor, conf = correct[answered], confidence[answered]
    n = len(cor)
    res.n_scored = n
    res.n_correct = nc = int(cor.sum())
    res.n_incorrect = ni = n - nc
    res.accuracy = nc / n if n else np.nan
    res.a = a = int((cor & conf).sum())
    res.b = b = int((~cor & conf).sum())
    res.c = c = int((cor & ~conf).sum())
    res.d = d = int((~cor & ~conf).sum())

    # TRIN: structural indicator only
    trin = max(a + b, c + d) / n if n else np.nan
    res.trin = Index("TRIN", trin, trin, trin, n,
                     "warning" if trin >= TRIN_CUT else "ok")

    # Fp = P(low confidence | correct)
    res.Fp = Index("Fp", n=nc)
    if nc >= MIN_N:
        fp = c / nc
        lo, hi = wilson_ci(c, nc, alpha)
        res.Fp.value, res.Fp.lo, res.Fp.hi = fp, lo, hi
        if fp >= FP_CUT:
            res.Fp.flag = "invalid" if lo > FP_GUARD else "indeterminate"
            res.reasons.append(f"Fp = {fp:.3f} (Wilson lower {lo:.3f})")
    else:
        res.Fp.flag = "not_estimable"
        res.reasons.append(f"Fp not estimable: {nc} correct items < {MIN_N}")

    # L = P(high confidence | incorrect)
    res.L = Index("L", n=ni)
    if ni >= MIN_N:
        lv = b / ni
        lo, hi = wilson_ci(b, ni, alpha)
        res.L.value, res.L.lo, res.L.hi = lv, lo, hi
        if lv >= L_CUT:
            res.L.flag = "invalid" if lo > L_GUARD else "indeterminate"
            res.reasons.append(f"L = {lv:.3f} (Wilson lower {lo:.3f}; "
                               f"{'certified' if lo > L_GUARD else 'too few errors to certify'})")
    else:
        res.L.flag = "not_estimable"
        res.reasons.append(f"L not estimable: {ni} errors < {MIN_N}")

    # RBS = Fp - (1 - L) = P(W|correct) - P(W|incorrect) = -Youden's J
    res.rbs = Index("RBS", n=n)
    if nc >= MIN_N and ni >= MIN_N:
        rbs = c / nc - d / ni
        lo, hi = newcombe_diff_ci(c, nc, d, ni, alpha)
        res.rbs.value, res.rbs.lo, res.rbs.hi = rbs, lo, hi
        res.youden_j = -rbs
        res.auroc_binary = (1 - rbs) / 2
        if rbs > 0:
            res.rbs.flag = "invalid" if lo > 0 else "indeterminate"
            res.reasons.append(f"RBS = {rbs:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    else:
        res.rbs.flag = "not_estimable"

    # r(confidence, correct) = phi
    res.r = Index("r", n=n)
    if n > 3 and conf.std() > 0 and cor.std() > 0:
        rv, pv = sp_stats.pearsonr(conf.astype(float), cor.astype(float))
        zr = np.arctanh(np.clip(rv, -0.999999, 0.999999))
        se = 1 / np.sqrt(n - 3)
        zc = sp_stats.norm.ppf(1 - alpha / 2)
        res.r.value, res.r.lo, res.r.hi = rv, np.tanh(zr - zc * se), np.tanh(zr + zc * se)
        res.r.flag = f"p={pv:.4f}"

    flags = [res.Fp.flag, res.L.flag, res.rbs.flag]
    if "invalid" in flags:
        res.tier = "Invalid"
    elif "indeterminate" in flags:
        res.tier = "Indeterminate"
    elif flags.count("not_estimable") >= 2 or res.rbs.flag == "not_estimable":
        res.tier = "Uncertifiable"
    else:
        res.tier = "Valid"
    res.tier_v02 = res.tier
    if require_signal and res.tier == "Valid" and not (res.rbs.hi < 0):
        res.tier = "Indeterminate"
        res.reasons.append(f"no detectable item sensitivity: RBS = {res.rbs.value:+.3f} "
                           f"[{res.rbs.lo:+.3f}, {res.rbs.hi:+.3f}] includes 0")
    return res


def is_nonanswer(answer, correct_answer) -> bool:
    """Frozen rule for unscorable outputs.

    A row is a non-answer if the recorded answer is a decline marker, or is a
    bare option letter (A-D) when the correct answer is not itself a letter
    (i.e. the parser captured a path/option choice rather than an answer).
    """
    a = str(answer).strip()
    ca = str(correct_answer).strip()
    if "DECLIN" in a.upper():
        return True
    bare = len(a.rstrip(".)")) == 1 and a.rstrip(".)").upper() in "ABCD"
    ca_is_letter = len(ca) == 1 and ca.upper() in "ABCD"
    return bare and not ca_is_letter
