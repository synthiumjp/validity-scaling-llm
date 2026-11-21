"""Collect every number quoted in the paper from the analysis outputs.

Run: python3 -I numbers.py <out_dir>
Reads cells.csv and oc_*.csv from <out_dir>; writes numbers.json there.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import validity_screen as vs

OUT = Path(sys.argv[1])
C = pd.read_csv(OUT / "cells.csv")
N = {}
CUTS = [50, 70, 80, 90, 95, 100]


def counts(S):
    return S.tier.value_counts().reindex(["Valid", "Indeterminate", "Invalid", "Uncertifiable"], fill_value=0).to_dict()


# ---------------- battery
B = C[C.ds == "battery"].set_index("unit")
N["battery"] = dict(
    n_models=len(B), n_responses=int(B.n_items.sum()), n_scored=int(B.n_scored.sum()),
    tiers=counts(B.reset_index()),
    members={t: sorted(B.index[B.tier == t]) for t in B.tier.unique()},
    reasons=B.reason.value_counts().to_dict(),
    lt35_errors=int((B.n_err < 35).sum()),
    valid_J_range=[B[B.tier == "Valid"].J.min(), B[B.tier == "Valid"].J.max()],
    valid_L_range=[B[B.tier == "Valid"].L.min(), B[B.tier == "Valid"].L.max()],
    flags={f: sorted(B.index[B["flags"].fillna("").str.contains(f)]) for f in
           ["low_coverage", "response_set_not_excluded", "fixed_responding"]},
    v03_to_v04=pd.crosstab(B.tier_v03, B.tier).stack().loc[lambda s: s > 0].to_dict(),
    rows={m: B.loc[m, ["n_err", "accuracy", "L", "L_lo", "L_hi", "Fp", "Fp_lo", "Fp_hi", "J", "J_lo", "J_hi",
                       "p_signal", "tier", "reason", "flags", "auroc_raw"]].to_dict() for m in B.index})
B6 = C[C.ds == "battery6"].set_index("unit")
N["battery6"] = dict(tiers=counts(B6.reset_index()),
                     differs={m: [B.tier[m], B6.tier[m]] for m in B.index if B.tier[m] != B6.tier[m]})

# ---------------- verbalized: atlas, mmlu500, yang basic_1s
for ds in ["atlas", "mmlu500", "mmlu500_no50"]:
    S = C[C.ds == ds]
    N[ds] = dict(n_models=S.unit.nunique(), n_scored=int(S[S.cut == 90].n_scored.sum()),
                 by_cut={int(k): counts(S[S.cut == k]) for k in CUTS},
                 reasons_by_cut={int(k): S[S.cut == k].reason.value_counts().to_dict() for k in CUTS},
                 auroc_raw=dict(mean=S.groupby("unit").auroc_raw.first().mean(),
                                min=S.groupby("unit").auroc_raw.first().min(),
                                max=S.groupby("unit").auroc_raw.first().max()),
                 v03_changes=int((S.tier != S.tier_v03).sum()), cells=len(S))
a5 = C[C.ds == "mmlu500"].set_index(["unit", "cut"]).tier
a5x = C[C.ds == "mmlu500_no50"].set_index(["unit", "cut"]).tier
N["mmlu500_50s_changes"] = {f"{u}@{int(k)}": [a5[(u, k)], a5x[(u, k)]] for (u, k) in a5.index if a5[(u, k)] != a5x[(u, k)]}

# Valid but weak: Valid cells with small J
for ds in ["atlas", "mmlu500"]:
    S = C[(C.ds == ds) & (C.tier == "Valid")]
    N[ds]["valid_J_lt_10"] = int((S.J < .10).sum())
    N[ds]["valid_cells"] = len(S)

# Replication across MMLU samples (20 models in both)
at = C[C.ds == "atlas"]; m5 = C[C.ds == "mmlu500"]
name_map = {}
for u in m5.unit.unique():
    key = u.lower().replace(" ", "-")
    cand = [a for a in at.unit.unique() if key.split("-validity")[0] in a.lower()]
    name_map[u] = cand[0] if len(cand) == 1 else None
N["name_map_mmlu500_to_atlas"] = name_map

Y = C[C.ds == "yang"]
Yb = Y[Y.method == "basic_1s"]
N["yang"] = dict(
    n_cells_all=int(Y[Y.cut == 90].shape[0]), n_methods=Y.method.nunique(),
    n_models=Y.unit.str.split("|").str[0].nunique(), n_benches=Y.unit.str.split("|").str[1].nunique(),
    n_scored_all=int(Y[Y.cut == 90].n_scored.sum()),
    basic=dict(cells=int(Yb[Yb.cut == 90].shape[0]), n_scored=int(Yb[Yb.cut == 90].n_scored.sum()),
               by_cut={int(k): counts(Yb[Yb.cut == k]) for k in CUTS},
               reasons_by_cut={int(k): Yb[Yb.cut == k].reason.value_counts().to_dict() for k in CUTS},
               auroc_mean=Yb.groupby("unit").auroc_raw.first().mean(),
               auroc_range=[Yb.groupby("unit").auroc_raw.first().min(), Yb.groupby("unit").auroc_raw.first().max()]),
    all_by_cut={int(k): counts(Y[Y.cut == k]) for k in CUTS},
)
# method vs model dependence at cut 90: share Valid by method and by model
y90 = Y[Y.cut == 90].assign(model=lambda d: d.unit.str.split("|").str[0], bench=lambda d: d.unit.str.split("|").str[1])
by_method = y90.groupby("method").tier.apply(lambda t: (t == "Valid").mean())
by_model = y90.groupby("model").tier.apply(lambda t: (t == "Valid").mean())
by_bench = y90.groupby("bench").tier.apply(lambda t: (t == "Valid").mean())
N["yang"]["valid90_by_method"] = dict(min=by_method.min(), max=by_method.max(),
                                      argmin=by_method.idxmin(), argmax=by_method.idxmax())
N["yang"]["valid90_by_model"] = dict(min=by_model.min(), max=by_model.max(),
                                     argmin=by_model.idxmin(), argmax=by_model.idxmax())
N["yang"]["valid90_by_bench"] = dict(min=by_bench.min(), max=by_bench.max(),
                                     argmin=by_bench.idxmin(), argmax=by_bench.idxmax())
inv90 = y90.groupby("method").tier.apply(lambda t: (t == "Invalid").mean())
N["yang"]["invalid90_by_method"] = dict(min=inv90.min(), max=inv90.max())
# raw AUROC of Valid vs non-Valid cells at 90
N["yang"]["auroc_by_tier90"] = y90.groupby("tier").auroc_raw.mean().to_dict()
N["yang"]["v03_changes_all"] = int((Y.tier != Y.tier_v03).sum())
N["yang"]["v03_to_v04_all"] = {f"{a}->{b}": int(v) for (a, b), v in
                              pd.crosstab(Y.tier_v03, Y.tier).stack().items() if v > 0 and a != b}

# ---------------- operating characteristics
oc = json.loads((OUT / "oc_numbers.json").read_text())
N["oc"] = oc
S = pd.read_csv(OUT / "oc_size.csv")
N["oc"]["size_max_where"] = S.loc[S.false_valid.idxmax()].to_dict()
P = pd.read_csv(OUT / "oc_power_valid.csv")
pick = lambda n, a, j, k=.95: float(P[(P.N == n) & (P.accuracy == a) & (P.J == j) & (P.keep_correct == k)].p_valid.iloc[0])
N["oc"]["power_examples"] = {f"N{n}_acc{a}_J{j}": pick(n, a, j) for n in [452, 1000, 1500, 3000]
                             for a in [.90, .95, .97] for j in [.05, .10, .20]}
N["oc"]["power_invalid_examples"] = {f"L{L}_e{e}": vs.power_invalid(L, e) for L in [1.0, .99, .98, .97, .96, .95]
                                     for e in [35, 50, 71, 100, 150, 500, 1000]}
N["oc"]["items_needed_90"] = {f"L{L}_acc{a}": vs.items_needed(a, L, .9) for L in [1.0, .99, .98, .97, .96, .95]
                              for a in [.80, .90, .95]}
AG = pd.read_csv(OUT / "oc_agreement.csv")
N["oc"]["agreement_low452"] = AG[AG.N == 452].nsmallest(5, "agree")[["model", "agree"]].values.tolist()
CM = pd.read_csv(OUT / "oc_cmh_battery.csv", index_col=0)
N["oc"]["cmh_valid_not_sig"] = sorted([m for m in CM.index if m in B.index and B.tier[m] == "Valid"
                                       and CM.p_one_sided[m] >= .025])

# Newcombe vs mid-p size, for the comparison in the text (452 items)
def newcombe_size(N_, acc, keep):
    ni = int(round(N_ * (1 - acc))); nc = N_ - ni
    pa = stats.binom.pmf(np.arange(nc + 1), nc, keep); pb = stats.binom.pmf(np.arange(ni + 1), ni, keep)
    tot = 0.0
    for a in np.nonzero(pa > 1e-12)[0]:
        for b in np.nonzero(pb > 1e-12)[0]:
            L, Fp = b / ni, (nc - a) / nc
            if L >= .95 or Fp >= .95:
                continue
            lo, hi = vs.newcombe_diff_ci(int(a), nc, int(b), ni)
            if lo > 0:
                tot += pa[a] * pb[b]
    return tot


grid = [(452, acc, k) for acc in [.978, .967, .956, .934, .889] for k in [.05, .10, .14, .20, .28, .40, .50, .60, .70, .80, .90, .95, .99]]
comp = [dict(N=n, errors=int(round(n * (1 - a))), keep=k, newcombe=newcombe_size(n, a, k),
             midp=vs.power_valid(n, a, k, k)) for n, a, k in grid]
CP = pd.DataFrame(comp)
CP.to_csv(OUT / "oc_size_compare.csv", index=False)
N["oc"]["size_compare"] = dict(newcombe_max=CP.newcombe.max(), midp_max=CP.midp.max())


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


(OUT / "numbers.json").write_text(json.dumps(clean(N), indent=1))
print(json.dumps(clean({k: N[k] for k in ["battery6", "mmlu500_50s_changes"]}), indent=1))
print(json.dumps(clean(N["yang"]), indent=1)[:3000])
print(json.dumps(clean(N["oc"]["size_compare"])), json.dumps(clean(N["atlas"]["by_cut"])))
