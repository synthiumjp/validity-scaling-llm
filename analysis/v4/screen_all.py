"""Screen every unit in every data set with validity-screen 0.4 (the installed package).

Run: python3 -I screen_all.py <battery_csvs> <atlas_raw_v2> <mmlu500> <yang_predictions> <out_dir>

Writes <out>/cells.csv: one row per unit x cut point, with the v0.4 tier, reason code,
flags, J and its interval, the mid-p for J > 0, L and Fp with intervals, raw AUROC,
and the v0.3 tier from the frozen legacy module for the comparison table.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import validity_screen as vs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

spec = importlib.util.spec_from_file_location("legacy03", HERE / "legacy" / "validity_screen_v03.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

BATT, ATLAS, MMLU, YANG, OUT = map(Path, sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=True)
CUTS = [50, 70, 80, 90, 95, 100]
assert vs.__version__ == "0.4.1", vs.__version__


def row_of(r, **meta):
    J = r.J
    return dict(meta, n_items=r.n_items, n_excluded=r.n_excluded, n_scored=r.n_scored,
                n_correct=r.n_correct, n_err=r.n_incorrect, accuracy=r.accuracy,
                a=r.a, b=r.b, c=r.c, d=r.d,
                tier=r.tier, reason=r.reason_code, flags="|".join(r.flags),
                J=J.value, J_lo=J.lo, J_hi=J.hi, p_signal=r.p_signal, p_inversion=r.p_inversion,
                L=r.L.value, L_lo=r.L.lo, L_hi=r.L.hi, Fp=r.Fp.value, Fp_lo=r.Fp.lo, Fp_hi=r.Fp.hi,
                auroc_binary=r.auroc_binary)


def legacy_tier(cor, keep):
    return legacy.screen(cor, keep).tier


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    return roc_auc_score(y, s) if 0 < y.mean() < 1 and np.std(s) > 0 else np.nan


rows = []

# ---------------- battery (binary KEEP/WITHDRAW)
for label, drop in [("battery", ("prospective",)), ("battery6", ())]:
    D = common.load_battery(BATT, drop_tracks=drop)
    for m, g in D.groupby("model"):
        r = vs.screen(g.correct.values, g.keep.values, answered=g.answered.values, model_name=m)
        s = g[g.answered == 1]
        rows.append(row_of(r, ds=label, unit=m, method="KEEP/WITHDRAW", cut=np.nan,
                           auroc_raw=auroc(s.correct, s.ordinal),
                           tier_v03=legacy_tier(s.correct.values, s.keep.values)))

# ---------------- MMLU data sets (verbalized 0-100)
def mmlu_rows(label, M):
    for m, g in M.groupby("model"):
        s = g[g.answered == 1]
        ar = auroc(s.correct, s.confidence)
        for k in CUTS:
            r = vs.screen(g.correct.values, g.confidence.values, answered=g.answered.values, cut=k,
                          model_name=m)
            rows.append(row_of(r, ds=label, unit=m, method="verbal 0-100", cut=k, auroc_raw=ar,
                               tier_v03=legacy_tier(s.correct.values, (s.confidence >= k).values)))


mmlu_rows("atlas", common.load_mmlu(ATLAS))
M5 = common.load_mmlu(MMLU)
mmlu_rows("mmlu500", M5)
M5x = M5.copy()
M5x.loc[M5x.confidence == 50, "answered"] = 0       # sensitivity: the nine 50s of unknown status
mmlu_rows("mmlu500_no50", M5x)

# ---------------- Yang et al. (2024), all methods
Y = common.load_yang(YANG)
for (bench, model, method), g in Y.groupby(["bench", "model", "method"]):
    s = g[g.answered == 1]
    if len(s) < 20 or s.correct.nunique() < 2:
        continue
    ar = auroc(s.correct, s.confidence)
    for k in CUTS:
        r = vs.screen(g.correct.values, g.confidence.values, answered=g.answered.values, cut=k,
                      model_name=model, benchmark_name=bench)
        rows.append(row_of(r, ds="yang", unit=f"{model}|{bench}", method=method, cut=k, auroc_raw=ar,
                           tier_v03=legacy_tier(s.correct.values, (s.confidence >= k).values)))

C = pd.DataFrame(rows)
C.to_csv(OUT / "cells.csv", index=False)
pd.set_option("display.width", 220)
print(C.groupby(["ds", "cut"], dropna=False).tier.value_counts().unstack(fill_value=0).to_string())
