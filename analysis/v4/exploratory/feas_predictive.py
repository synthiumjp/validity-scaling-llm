"""Feasibility check: does the screen verdict on one half of a cell's items predict whether
abstaining at that cut point improves accuracy on the other half?

Yang et al. (2024), all 17 methods, cut points 70/80/90/95. For each cell and cut:
split items at random into A and B; screen A; on B compute the gain in accuracy from
answering only kept items (accuracy among kept minus overall accuracy) and coverage.
Repeated over 5 random splits. Run: python3 -I feas_predictive.py <yang_predictions> <out_csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import validity_screen as vs

sys.path.append(str(Path(__file__).resolve().parent))
import common  # noqa: E402

Y = common.load_yang(sys.argv[1])
Y = Y[Y.answered == 1]
rng = np.random.default_rng(7)
rows = []
for (bench, model, method), g in Y.groupby(["bench", "model", "method"]):
    cor, conf = g.correct.values, g.confidence.values
    n = len(cor)
    if n < 100 or cor.mean() in (0, 1):
        continue
    for k in [70, 80, 90, 95]:
        for rep in range(5):
            idx = rng.permutation(n)
            A, B = idx[: n // 2], idx[n // 2:]
            r = vs.screen(cor[A], conf[A], cut=k)
            keepB = conf[B] >= k
            cov = keepB.mean()
            gain = (cor[B][keepB].mean() - cor[B].mean()) if keepB.any() else np.nan
            rows.append(dict(bench=bench, model=model, method=method, cut=k, rep=rep, n=n,
                             tierA=r.tier, reasonA=r.reason_code, JA=r.J.value,
                             coverageB=cov, gainB=gain))
R = pd.DataFrame(rows)
R.to_csv(sys.argv[2], index=False)
pd.set_option("display.width", 200)
print(R.groupby("tierA").agg(n=("gainB", "size"), gain_mean=("gainB", "mean"),
                             gain_median=("gainB", "median"), cov_mean=("coverageB", "mean"),
                             share_gain_gt_2pts=("gainB", lambda x: (x > .02).mean())).round(3))
print(R.groupby(["cut", "tierA"]).gainB.mean().unstack().round(3))
