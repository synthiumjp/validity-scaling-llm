"""Consolidate cross-benchmark results (atlas MMLU, MMLU-500, Yang et al.) into
cross_numbers.json. Run after run_all.sh. Usage: python3 -I cross_benchmark.py <outdir>
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

OUT = Path(sys.argv[1])
X = {}
CUTS = [50, 70, 80, 90, 95, 100]
BATTERY = {
    "Claude Haiku 4.5": "claude-haiku-4-5", "Opus 4.6": "claude-opus-4-6", "Sonnet 4.6": "claude-sonnet-4-6",
    "DeepSeek V3.2": "deepseek-v3.2", "DeepSeek-R1": "deepseek-r1", "GLM-5": "glm-5",
    "GPT-5.4": "gpt-5.4-2026", "GPT-5.4 mini": "gpt-5.4-mini", "GPT-5.4 nano": "gpt-5.4-nano",
    "Gemini 2.5 Flash": "gemini-2.5-flash", "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 3 Flash": "gemini-3-flash", "Gemini 3.1 Pro": "gemini-3.1-pro",
    "Gemma 3 12B": "gemma-3-12b", "Gemma 3 1B": "gemma-3-1b", "Gemma 3 27B": "gemma-3-27b",
    "Qwen 235B": "qwen3-235b", "Qwen 80B Inst": "80b-a3b-instruct", "Qwen 80B Think": "80b-a3b-thinking",
    "Qwen Coder 480B": "coder-480b",
}


def find(df, key):
    m = df[df.model.str.contains(re.escape(key), case=False)]
    if key == "gpt-5.4-2026":
        m = m[~m.model.str.contains("mini|nano", case=False)]
    return m.iloc[0] if len(m) == 1 else None


P = pd.read_csv(OUT / "per_model.csv").set_index("model")
A = pd.read_csv(OUT / "atlas" / "mmlu_per_model.csv")
B = pd.read_csv(OUT / "mmlu20" / "mmlu_per_model.csv")
for lbl, D in [("atlas", A), ("mmlu500", B)]:
    X[f"{lbl}_n_models"] = len(D)
    X[f"{lbl}_partial_runs"] = int((D.n_raw < 1500).sum()) if lbl == "atlas" else int((D.n_raw < 500).sum())
    X[f"{lbl}_partial_sizes"] = sorted(D.loc[D.n_raw < (1500 if lbl == "atlas" else 500), "n_raw"].astype(int).tolist())
    X[f"{lbl}_scored_total"] = int(D.n.sum())
    X[f"{lbl}_excluded_total"] = int(D.n_excluded.sum())
    X[f"{lbl}_answer_X_total"] = int(D.n_answer_X.sum())
    for c in CUTS:
        X[f"{lbl}_tiers_c{c}"] = D[f"c{c}_tier"].value_counts().to_dict()
        X[f"{lbl}_tiers_c{c}_v02"] = D[f"c{c}_tier_v02"].value_counts().to_dict()
    X[f"{lbl}_tiers_gt50"] = D.gt50_tier.value_counts().to_dict()
    X[f"{lbl}_share_ge90_median"] = D.share_ge90.median()
    X[f"{lbl}_auroc_range"] = [D.auroc_cont.min(), D.auroc_cont.max()]
    X[f"{lbl}_auroc_mean"] = D.auroc_cont.mean()
    X[f"{lbl}_auroc_ci_incl_05"] = D.loc[D.auroc_lo <= 0.5, "model"].tolist()
    X[f"{lbl}_best_cut_valid"] = {c: int((D[f"c{c}_tier"] == "Valid").sum()) for c in CUTS}
    # models Valid at some cut
    anyv = np.zeros(len(D), bool)
    for c in CUTS:
        anyv |= (D[f"c{c}_tier"] == "Valid").values
    X[f"{lbl}_valid_at_some_cut"] = int(anyv.sum())
    X[f"{lbl}_never_valid"] = D.loc[~anyv, ["model", "auroc_cont", "auroc_lo", "auroc_hi"]].round(3).values.tolist()

A["k"] = A.model.str.replace(r"[@-]20\d\d.*$|@default$", "", regex=True)
B["k"] = B.model.str.replace(r"[@-]20\d\d.*$|@default$", "", regex=True)
M = A.merge(B, on="k", suffixes=("_a", "_b"))
X["retest_n"] = len(M)
X["retest_r_auroc"] = stats.pearsonr(M.auroc_cont_a, M.auroc_cont_b)[0]
for c in CUTS:
    X[f"retest_agree_c{c}"] = float((M[f"c{c}_tier_a"] == M[f"c{c}_tier_b"]).mean())

rows = []
for name, key in BATTERY.items():
    a, b = find(A, key), find(B, key)
    rows.append(dict(model=name, battery_tier=P.loc[name, "tier"], battery_tier_v02=P.loc[name, "tier_v02"],
                     battery_r=P.loc[name, "r"],
                     atlas_auroc=a.auroc_cont, atlas_lo=a.auroc_lo, atlas_hi=a.auroc_hi,
                     atlas_c50=a.c50_tier, atlas_c90=a.c90_tier,
                     mmlu500_auroc=b.auroc_cont, mmlu500_c50=b.c50_tier, mmlu500_c90=b.c90_tier))
C = pd.DataFrame(rows)
C.to_csv(OUT / "battery_vs_mmlu.csv", index=False)
X["battery_vs_mmlu"] = C.round(3).to_dict(orient="records")
X["corr_battery_r_atlas_auroc"] = stats.spearmanr(C.battery_r, C.atlas_auroc)[0]
X["corr_battery_r_mmlu500_auroc"] = stats.spearmanr(C.battery_r, C.mmlu500_auroc)[0]

Y = pd.read_csv(OUT / "yang" / "yang_basic_1s_cells.csv")
X["yang_cells"] = len(Y)
X["yang_frac_cells"] = int(Y.frac_labels.sum())
X["yang_items_total"] = int(Y.n.sum())
for c in ["c50", "c70", "c80", "c90", "c95", "c100"]:
    X[f"yang_tiers_{c}"] = Y[f"{c}_tier"].value_counts().to_dict()
    X[f"yang_tiers_{c}_v02"] = Y[f"{c}_tier_v02"].value_counts().to_dict()
X["yang_auroc_mean"] = Y.auroc_cont.mean()
X["yang_auroc_range"] = [Y.auroc_cont.min(), Y.auroc_cont.max()]
X["yang_invalid_c90_by_model"] = Y[Y.c90_tier == "Invalid"].model.value_counts().to_dict()
q = Y.dropna(subset=["c90_L", "auroc_cont"])
X["yang_rho_L90_auroc"] = stats.spearmanr(q.c90_L, q.auroc_cont)[0]
q7 = Y[Y.model == "qwen1.5-7b-chat"]
X["yang_qwen7b"] = dict(auroc_mean=q7.auroc_cont.mean(), auroc_max=q7.auroc_cont.max(), conf_sd_mean=q7.conf_sd.mean(),
                        n_distinct_max=int(q7.n_distinct.max()), invalid_c90=int((q7.c90_tier == "Invalid").sum()),
                        invalid_c50=int((q7.c50_tier == "Invalid").sum()))
g7 = Y[Y.model == "gemma1.1-7b-it"]
X["yang_gemma7b"] = dict(auroc_mean=g7.auroc_cont.mean(), invalid_c90=int((g7.c90_tier == "Invalid").sum()),
                         reasons=g7.c90_reasons.str[:60].tolist() if "c90_reasons" in g7 else None)
S = pd.read_csv(OUT / "yang" / "yang_by_method.csv")
X["yang_methods"] = len(S)
X["yang_methods_inval90_range"] = [S.inval90.min(), S.inval90.max()]
X["yang_methods_inval50_range"] = [S.inval50.min(), S.inval50.max()]
try:
    Yg = pd.read_csv(OUT / "yang_gt0" / "yang_basic_1s_cells.csv")
    X["yang_gt0_tiers_c90"] = Yg.c90_tier.value_counts().to_dict()
except FileNotFoundError:
    pass


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


(OUT / "cross_numbers.json").write_text(json.dumps(clean(X), indent=1))
print(json.dumps(clean({k: v for k, v in X.items() if k != "battery_vs_mmlu"}), indent=1))
