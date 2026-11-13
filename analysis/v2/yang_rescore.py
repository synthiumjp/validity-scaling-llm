"""Yang et al. (2024) released predictions, screened with validity_screen v0.2.
Data: predictions/y_pred/<bench>/<model>/<method>.npy (confidence 0-1) and
      predictions/y_true/<bench>/<model>/<method>.npy (correct 0/1).
Run: python3 -I yang_rescore.py <predictions_dir> <out_dir>
Primary method: basic_1s (Yang et al.'s default verbalized prompt). All other
methods are reported as sensitivity. Correctness: y_true is fractional partial
credit for trivia_qa and truthful_qa-mc2; it is binarised at >= 0.5 (sensitivity:
> 0). Confidence cut points c in CUTS (confidence >= c is "high"); no median split.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validity_screen import screen  # noqa: E402

BASE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TCUT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
CUTS = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for fp in sorted((BASE / "y_pred").glob("*/*/*.npy")):
    bench, model, method = fp.parts[-3], fp.parts[-2], fp.stem
    ft = BASE / "y_true" / bench / model / f"{method}.npy"
    if not ft.exists():
        continue
    p = np.load(fp, allow_pickle=False).astype(float)
    t = np.load(ft, allow_pickle=False).astype(float)
    if p.shape != t.shape:
        rows.append(dict(bench=bench, model=model, method=method, note="shape mismatch"))
        continue
    ok = np.isfinite(p) & np.isfinite(t)
    frac = bool(np.any((t[ok] > 0) & (t[ok] < 1)))
    p, t = p[ok], (t[ok] >= TCUT).astype(int)
    row = dict(bench=bench, model=model, method=method, n=len(t), frac_labels=frac, n_dropped=int((~ok).sum()),
               accuracy=t.mean() if len(t) else np.nan, errors=int((t == 0).sum()),
               conf_median=float(np.median(p)) if len(p) else np.nan,
               conf_sd=float(p.std()) if len(p) else np.nan, n_distinct=int(len(np.unique(p))),
               auroc_cont=roc_auc_score(t, p) if len(t) and 0 < t.mean() < 1 and p.std() > 0 else np.nan)
    if len(t) < 20 or not (0 < t.mean() < 1):
        rows.append(row)
        continue
    for thr in CUTS:
        lbl = f"c{int(round(thr*100))}"
        hi = (p >= thr).astype(int)
        r = screen(t, hi)
        row.update({f"{lbl}_tier": r.tier, f"{lbl}_tier_v02": r.tier_v02, f"{lbl}_L": r.L.value, f"{lbl}_Llo": r.L.lo, f"{lbl}_Fp": r.Fp.value,
                    f"{lbl}_rbs": r.rbs.value, f"{lbl}_r": r.r.value, f"{lbl}_aurocbin": r.auroc_binary,
                    f"{lbl}_keep": hi.mean(), f"{lbl}_reasons": "; ".join(r.reasons)})
    rows.append(row)

D = pd.DataFrame(rows)
D.to_csv(OUT / "yang_all_cells.csv", index=False)
P = D[D.method == "basic_1s"].copy()
P.to_csv(OUT / "yang_basic_1s_cells.csv", index=False)
pd.set_option("display.width", 250)
print("== basic_1s: cells", len(P), "| models", P.model.nunique(), "| benches", P.bench.nunique(),
      "| cells with fractional labels", int(P.frac_labels.sum()))
for c in ["c50", "c70", "c80", "c90", "c95", "c100"]:
    print(c, P[f"{c}_tier"].value_counts().to_dict())
print(pd.crosstab(P.model, P.c90_tier).to_string())
print("mean continuous AUROC:", round(P.auroc_cont.mean(), 3), "range", round(P.auroc_cont.min(), 3), round(P.auroc_cont.max(), 3))
M = P.groupby("model").agg(acc=("accuracy", "mean"), auroc=("auroc_cont", "mean"), auroc_max=("auroc_cont", "max"),
                            conf_sd=("conf_sd", "mean"), n_distinct_max=("n_distinct", "max"),
                            inval90=("c90_tier", lambda s: (s == "Invalid").sum()),
                            inval50=("c50_tier", lambda s: (s == "Invalid").sum()))
print(M.round(3).to_string()); M.to_csv(OUT / "yang_basic_1s_by_model.csv")
S = D.dropna(subset=["auroc_cont"]).groupby("method").agg(cells=("bench", "size"), auroc=("auroc_cont", "mean"),
    inval50=("c50_tier", lambda s: (s == "Invalid").mean()), inval90=("c90_tier", lambda s: (s == "Invalid").mean()))
print(S.round(3).to_string()); S.to_csv(OUT / "yang_by_method.csv")
