"""Feasibility check: does a screen verdict on benchmark X predict the accuracy gain from
abstaining at the same cut on benchmark Y (same model, same prompting method)?

Yang et al. (2024), 17 methods, cuts 80/90/95. Gain on Y = accuracy among kept items minus
overall accuracy. Run: python3 -I feas_transfer.py <out_dir>
"""
import sys
import numpy as np
import pandas as pd

OUT = sys.argv[1]
C = pd.read_csv(f"{OUT}/cells.csv")
Y = C[(C.ds == "yang") & C.cut.isin([80, 90, 95])].copy()
Y[["model", "bench"]] = Y.unit.str.split("|", expand=True)
# gain on each cell from its own table: acc among kept - overall acc
kept = Y.a + Y.b
Y["gain"] = np.where(kept > 0, Y.a / kept.where(kept > 0) - Y.n_correct / Y.n_scored, np.nan)
Y["coverage"] = kept / Y.n_scored
rows = []
for (model, method, cut), g in Y.groupby(["model", "method", "cut"]):
    g = g.set_index("bench")
    for x in g.index:
        for y in g.index:
            if x != y:
                rows.append(dict(model=model, method=method, cut=cut, X=x, Y=y, tierX=g.tier[x],
                                 tierY=g.tier[y], gainY=g.gain[y], covY=g.coverage[y]))
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("pairs:", len(T))
print(T.groupby("tierX").agg(n=("gainY", "size"), gain_mean=("gainY", "mean"),
                             gain_gt_2pts=("gainY", lambda x: (x > .02).mean()),
                             validY=("tierY", lambda t: (t == "Valid").mean())).round(3))
# agreement of verdicts across benchmarks
print("P(Valid on Y | Valid on X) =", round((T[T.tierX == "Valid"].tierY == "Valid").mean(), 3),
      " P(Valid on Y | not Valid on X) =", round((T[T.tierX != "Valid"].tierY == "Valid").mean(), 3))
