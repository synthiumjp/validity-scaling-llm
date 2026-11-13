"""LaTeX tables + extra numbers for 17714 v2. Run: python tables_v2.py <outdir> <paperdir> (see run_all.sh)"""
import json, sys, runpy
from pathlib import Path
import numpy as np, pandas as pd

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
OUT, PAP = Path(sys.argv[1]), Path(sys.argv[2])
P = pd.read_csv(OUT / "per_model.csv")


def f3(x, sign=False):
    if pd.isna(x):
        return "--"
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    s = s.replace("0.", ".", 1) if abs(x) < 1 else s
    return s.replace("-", "$-$")


def ci(lo, hi, sign=False):
    return "" if pd.isna(lo) else f"\\,[{f3(lo, sign)}, {f3(hi, sign)}]"


order = {"Invalid": 0, "Indeterminate": 1, "Valid": 2}
Q = P.assign(o=P.tier.map(order)).sort_values(["o", "r"], ascending=[True, False])
lines = []
for _, r in Q.iterrows():
    lines.append(" & ".join([
        r.model, str(int(r.n_incorrect)), f3(r.accuracy),
        f3(r.L) + ci(r.L_lo, r.L_hi), f3(r.Fp) + ci(r.Fp_lo, r.Fp_hi),
        (f3(r.rbs, True) + ci(r.rbs_lo, r.rbs_hi, True)) if not pd.isna(r.rbs) else "n.e.",
        f3(r.r, True), f3(r.trin), r.tier + ("" if r.tier == r.tier_v02 else f" ({r.tier_v02})")]) + " \\\\")
(PAP / "tab_battery.tex").write_text("\n".join(lines) + "\n")

# synthetic table: stochastic policies (synthetic_sim.json from synthetic_run.py)
SIM = json.loads((OUT / "synthetic_sim.json").read_text())
syn = []
pct = lambda v: f"{100 * v:.0f}"
for a, b in zip(SIM["40"], SIM["150"]):
    nm = a["name"].replace("%", "\\%")
    syn.append(f"{nm} & {pct(a['p_keep_correct'])} / {pct(a['p_keep_incorrect'])} & "
               f"{pct(a['Valid'])} & {pct(a['Indeterminate'])} & {pct(a['Invalid'])} & "
               f"{pct(b['Valid'])} & {pct(b['Indeterminate'])} & {pct(b['Invalid'])} \\\\")
(PAP / "tab_synthetic.tex").write_text("\n".join(syn) + "\n")

X = {}
tw = P[P.trin >= 0.95]
X["trin_warn_models"] = tw.model.tolist()
X["trin_warn_valid"] = int((tw.tier == "Valid").sum())
X["trin_warn_valid_r_range"] = [tw[tw.tier == "Valid"].r.min(), tw[tw.tier == "Valid"].r.max()]
X["n_valid_sig"] = int(((P.tier == "Valid") & (P.r_lo > 0)).sum())
X["valid_not_sig"] = P[(P.tier == "Valid") & ~(P.r_lo > 0)].model.tolist()
X["errors_range"] = [int(P.n_incorrect.min()), int(P.n_incorrect.max())]
X["models_lt35"] = P[P.n_incorrect < 35].model.tolist()
X["acc_range"] = [P.accuracy.min(), P.accuracy.max()]
X["L_range_valid"] = [P[P.tier == "Valid"].L.min(), P[P.tier == "Valid"].L.max()]
X["Fp_max_nonR1"] = P[P.model != "DeepSeek-R1"].Fp.max()
X["Fp_max_nonR1_model"] = P.loc[P[P.model != "DeepSeek-R1"].Fp.idxmax(), "model"]
X["rbs_all_negative_except"] = P[P.rbs > 0].model.tolist()
X["selacc"] = P.set_index("model").loc[["Sonnet 4.6", "DeepSeek-R1", "Gemini 3.1 Pro", "Qwen 80B Think", "Claude Haiku 4.5"],
                                       ["selacc_100", "selacc_50", "selacc_30", "selacc_10"]].round(4).to_dict(orient="index")
X["auroc_ord"] = P.set_index("model").auroc_ordinal.round(3).to_dict()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


(OUT / "numbers_tables.json").write_text(json.dumps(clean(X), indent=1))
print(json.dumps(clean({k: v for k, v in X.items() if k != "auroc_ord"}), indent=1))
print(open(PAP / "tab_battery.tex").read()); print(open(PAP / "tab_synthetic.tex").read())
