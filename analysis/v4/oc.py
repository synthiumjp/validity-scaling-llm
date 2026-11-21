"""Operating characteristics of the v0.4 screen.

Run: python3 -I oc.py <out_dir> <battery_csvs> <atlas_raw_v2>

1. False-Valid rate (exact) by items, accuracy and KEEP rate, under no signal.
2. Power of a Valid verdict (exact) for J = .05, .10, .20.
3. Power to certify blanket confidence (exact) and items needed for 90% power.
4. Cut-point multiplicity: permutation of correctness within each atlas model.
5. Verdict stability: probability that two independent administrations agree, by items
   (parametric, from each battery model's observed rates).
6. Information lost by binarising 0-100 confidence (atlas).
7. Battery: association stratified by track (Cochran-Mantel-Haenszel).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import validity_screen as vs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT, BATT, ATLAS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NUM = {}
rng = np.random.default_rng(20261121)

# ---------------- 1. size
size = []
for N in [226, 452, 1000, 1500]:
    for acc in [.70, .80, .90, .95, .97]:
        if N * (1 - acc) < vs.MIN_N:
            continue
        for k in [.50, .80, .90, .95, .99]:
            size.append(dict(N=N, accuracy=acc, keep=k, false_valid=vs.power_valid(N, acc, k, k)))
S = pd.DataFrame(size)
S.to_csv(OUT / "oc_size.csv", index=False)
NUM["size_max"] = S.false_valid.max()
NUM["size_range"] = [S.false_valid.min(), S.false_valid.max()]
NUM["size_cells"] = len(S)

# ---------------- 1b. size over the full grid (keep rates .02 to .995, error counts from 10)
full = []
for N_ in [226, 452, 1000, 1500, 5000]:
    for e in [10, 11, 12, 15, 20, 30, 50, 100, 200]:
        acc = 1 - e / N_
        for k in [.02, .05, .10, .14, .20, .28, .40, .50, .60, .70, .80, .90, .95, .99, .995]:
            full.append(dict(N=N_, errors=e, keep=k, false_valid=vs.power_valid(N_, acc, k, k)))
F = pd.DataFrame(full)
F.to_csv(OUT / "oc_size_full.csv", index=False)
NUM["size_full_max"] = F.false_valid.max()
NUM["size_full_max_where"] = F.loc[F.false_valid.idxmax()].to_dict()
NUM["size_full_share_gt_025"] = float((F.false_valid > .025).mean())
NUM["size_full_max_keep_ge_50"] = F[F.keep >= .5].false_valid.max()
NUM["size_full_max_keep_ge_50_N_le_1500"] = F[(F.keep >= .5) & (F.N <= 1500)].false_valid.max()
NUM["size_full_max_by_N"] = {int(n): g.false_valid.max() for n, g in F.groupby("N")}

# ---------------- 1c. false Invalid: certifying blanket confidence when the true kept-error rate is below .95
fi = []
for L in [.90, .92, .93, .94, .949]:
    for e in [35, 53, 100, 140, 200, 500, 1000]:
        fi.append(dict(true_L=L, errors=e, p_false_invalid=vs.power_invalid(L, e)))
FI = pd.DataFrame(fi)
FI.to_csv(OUT / "oc_false_invalid.csv", index=False)
NUM["false_invalid_max_by_L"] = {str(L): max(vs.power_invalid(L, e) for e in range(35, 1001))
                                 for L in [.90, .92, .93, .94]}

# ---------------- 2. power of Valid
pw = []
for N in [226, 452, 1000, 1500, 3000]:
    for acc in [.80, .90, .95, .97]:
        for kc in [.90, .95]:
            for J in [.05, .10, .20]:
                pw.append(dict(N=N, accuracy=acc, keep_correct=kc, J=J,
                               p_valid=vs.power_valid(N, acc, kc, kc - J)))
P = pd.DataFrame(pw)
P.to_csv(OUT / "oc_power_valid.csv", index=False)

# ---------------- 3. power to certify blanket confidence
pi = []
for L in [1.0, .99, .98, .97, .96, .95]:
    for e in [35, 50, 75, 100, 150, 200, 300, 500, 1000]:
        pi.append(dict(true_L=L, errors=e, p_certify=vs.power_invalid(L, e)))
PI = pd.DataFrame(pi)
PI.to_csv(OUT / "oc_power_invalid.csv", index=False)
need = []
for L in [1.0, .99, .98, .97, .96, .95]:
    for acc in [.80, .90, .95]:
        need.append(dict(true_L=L, accuracy=acc, items_90=vs.items_needed(acc, L, .90)))
pd.DataFrame(need).to_csv(OUT / "oc_items_needed.csv", index=False)
NUM["errors_needed_integer"] = [vs.errors_needed(k) for k in range(8)]

# ---------------- 4. cut-point multiplicity (atlas, permutation within model)
A = common.load_mmlu(ATLAS)
A = A[A.answered == 1]
CUTS = [50, 70, 80, 90, 95, 100]
B = 200
mult = []
for m, g in A.groupby("model"):
    cor, conf = g.correct.values, g.confidence.values
    any_valid, per_cut = 0, np.zeros(len(CUTS))
    for _ in range(B):
        pc = rng.permutation(cor)
        v = [vs.screen(pc, conf, cut=k).tier == "Valid" for k in CUTS]
        any_valid += any(v)
        per_cut += v
    mult.append(dict(model=m, any_cut=any_valid / B, **{f"c{k}": x / B for k, x in zip(CUTS, per_cut)}))
MU = pd.DataFrame(mult)
MU.to_csv(OUT / "oc_multiplicity_atlas.csv", index=False)
NUM["multiplicity_any_mean"] = MU.any_cut.mean()
NUM["multiplicity_any_max"] = MU.any_cut.max()
NUM["multiplicity_single_cut_mean"] = MU[[f"c{k}" for k in CUTS]].values.mean()

# ---------------- 5. two-administration agreement (parametric)
D = common.load_battery(BATT)
Ds = D[D.answered == 1]
agree = []
SIMS = 1000
for m, g in Ds.groupby("model"):
    acc = g.correct.mean()
    kc = g.keep[g.correct == 1].mean()
    kw = g.keep[g.correct == 0].mean() if (g.correct == 0).any() else 1.0
    for N in [226, 452, 1000, 2000, 5000]:
        same = 0
        for _ in range(SIMS):
            t = []
            for __ in range(2):
                cor = rng.random(N) < acc
                keep = np.where(cor, rng.random(N) < kc, rng.random(N) < kw)
                t.append(vs.screen(cor, keep).tier)
            same += t[0] == t[1]
        agree.append(dict(model=m, N=N, agree=same / SIMS))
AG = pd.DataFrame(agree)
AG.to_csv(OUT / "oc_agreement.csv", index=False)
NUM["agreement_by_N"] = {int(N): dict(mean=g.agree.mean(), n_ge_90=int((g.agree >= .9).sum()))
                         for N, g in AG.groupby("N")}

# ---------------- 6. binarisation loss (atlas)
loss = []
for m, g in A.groupby("model"):
    raw = roc_auc_score(g.correct, g.confidence)
    for k in CUTS:
        hi = (g.confidence >= k).astype(int)
        b = roc_auc_score(g.correct, hi) if hi.nunique() > 1 else 0.5
        loss.append(dict(model=m, cut=k, auroc_raw=raw, auroc_binary=b, loss=raw - b))
LO = pd.DataFrame(loss)
LO.to_csv(OUT / "oc_binarisation_loss.csv", index=False)
NUM["binarisation_loss_by_cut"] = {int(k): dict(mean=g.loss.mean(), max=g.loss.max())
                                   for k, g in LO.groupby("cut")}

# ---------------- 7. battery: stratified by track (CMH, one-sided for J > 0)
cmh = {}
for m, g in Ds.groupby("model"):
    tabs = []
    for t, h in g.groupby("track"):
        a = int(((h.correct == 1) & (h.keep == 1)).sum()); c = int(((h.correct == 1) & (h.keep == 0)).sum())
        b = int(((h.correct == 0) & (h.keep == 1)).sum()); d = int(((h.correct == 0) & (h.keep == 0)).sum())
        if (a + c) and (b + d) and (a + b) and (c + d):
            tabs.append([[a, c], [b, d]])
    if len(tabs) >= 2:
        from statsmodels.stats.contingency_tables import StratifiedTable
        st = StratifiedTable([np.array(x) for x in tabs])
        res = st.test_null_odds(correction=False)
        z = np.sign(np.log(st.oddsratio_pooled)) * np.sqrt(res.statistic)
        cmh[m] = dict(or_mh=st.oddsratio_pooled, p_one_sided=float(stats.norm.sf(z)), strata=len(tabs))
pd.DataFrame(cmh).T.to_csv(OUT / "oc_cmh_battery.csv")
NUM["cmh"] = cmh


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


(OUT / "oc_numbers.json").write_text(json.dumps(clean(NUM), indent=1))
print(json.dumps(clean({k: v for k, v in NUM.items() if k != "cmh"}), indent=1))
