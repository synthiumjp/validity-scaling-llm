"""MMLU verbalized-confidence screen with validity_screen v0.2 (v2.1 analysis).
Run: python3 -I mmlu_rescore.py <item_level_dir> <out_dir> ['{"gpt-oss-120b": 50}']

Per model:
  * Step 1: rows with unparsed answers ("X") or unparsed confidence are excluded;
    an optional JSON map names parser-default confidence values to treat as missing.
  * continuous Type 2 AUROC on raw confidence, with a 2,000-draw bootstrap 95% CI;
  * screen verdicts at fixed cut points (confidence >= c is "high"), c in CUTS;
  * at c = 50, the opposite tie direction (confidence > 50) as a sensitivity check.
The per-model median split used in an earlier draft is not used: with heavily tied
confidence its verdict is decided by the tie convention, and an exact 50/50 split
puts Fp on the .50 threshold by construction.
"""
import json, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validity_screen import screen  # noqa: E402

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
IMPUTED = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
CUTS = [50, 70, 80, 90, 95, 100]
OUT.mkdir(parents=True, exist_ok=True)


def boot_auc(y, s, name, B=2000):
    """Bootstrap CI; seeded per model so results do not depend on file order."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    n, vals = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if 0 < y[i].mean() < 1:
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


rows = []
for f in sorted(IN.glob("*.csv")):
    d = pd.read_csv(f).drop_duplicates(subset=["model", "item_id"], keep="first")
    name = str(d.model.iloc[0])
    n_raw = len(d)
    ans_ok = ~d.answer.astype(str).str.strip().str.upper().isin(["X", "", "NAN", "NONE"])
    conf = pd.to_numeric(d.confidence, errors="coerce")
    n_imputed = 0
    for key, val in IMPUTED.items():
        if key in name:
            n_imputed = int((conf == val).sum())
            conf = conf.mask(conf == val)
    ok = (ans_ok & conf.notna()).values
    cor = d.is_correct.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int).values[ok]
    c = conf.values[ok].astype(float)
    lo, hi = boot_auc(cor, c, name)
    row = dict(model=name, n_raw=n_raw, n=len(cor), n_excluded=int((~ok).sum()), n_imputed=n_imputed,
               n_answer_X=int((~ans_ok).sum()), accuracy=cor.mean(), errors=int((cor == 0).sum()),
               conf_median=float(np.median(c)), share_ge90=float((c >= 90).mean()),
               share_eq100=float((c == 100).mean()), n_distinct=int(len(np.unique(c))),
               auroc_cont=roc_auc_score(cor, c), auroc_lo=lo, auroc_hi=hi)
    for cut in CUTS:
        r = screen(cor, (c >= cut).astype(int))
        row.update({f"c{cut}_tier": r.tier, f"c{cut}_tier_v02": r.tier_v02, f"c{cut}_L": r.L.value, f"c{cut}_Llo": r.L.lo, f"c{cut}_Fp": r.Fp.value,
                    f"c{cut}_r": r.r.value, f"c{cut}_keep": (c >= cut).mean()})
    r = screen(cor, (c > 50).astype(int))
    row["gt50_tier"] = r.tier
    rows.append(row)
M = pd.DataFrame(rows).set_index("model")
M.to_csv(OUT / "mmlu_per_model.csv")
pd.set_option("display.width", 250)
print(M[["n", "n_excluded", "accuracy", "errors", "share_ge90", "auroc_cont", "auroc_lo", "auroc_hi"]
        + [f"c{c}_tier" for c in CUTS] + ["gt50_tier"]].round(3).to_string())
for c in CUTS:
    print(c, M[f"c{c}_tier"].value_counts().to_dict())
print("gt50", M.gt50_tier.value_counts().to_dict())
