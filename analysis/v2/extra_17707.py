"""17707-specific recomputation on rescored data (non-answers excluded).
Run: VS_DROP_TRACKS=prospective python extra_17707.py data/csvs analysis/v2/out (see run_all.sh)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.argv_saved = list(sys.argv)
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
src = (here / "rescore.py").read_text()
load_part = src.split("# ---------------------------------------------------------------- per model")[0]
ns = {"__file__": str(here / "rescore.py")}
exec(load_part, ns)          # builds D (all rows) and S (answered rows) only
D, S = ns["D"], ns["S"]
OUT = Path(sys.argv_saved[2])
P = pd.read_csv(OUT / "per_model.csv").set_index("model")
X = {}

# R1 KEEP x BET contingency, corrected and as-published (all rows)
for lbl, frame in [("corrected", S), ("all_rows", D)]:
    g = frame[frame.model == "DeepSeek-R1"]
    X[f"R1_contingency_{lbl}"] = {f"{'KEEP' if k else 'WD'}+{'BET' if b else 'NOBET'}": int(v)
                                  for (k, b), v in g.groupby(["keep", "bet"]).size().items()}
    h = frame[frame.model == "Claude Haiku 4.5"]
    X[f"Haiku_contingency_{lbl}"] = {f"{'KEEP' if k else 'WD'}+{'BET' if b else 'NOBET'}": int(v)
                                     for (k, b), v in h.groupby(["keep", "bet"]).size().items()}

# 17707 Tier 1 rules (no CI guards): RBS>0, L>=.95, F>=.50, Fp>=.50
t1 = (P.rbs > 0) | (P.L >= 0.95) | (P.F >= 0.50) | (P.Fp >= 0.50)
X["tier1_17707_rules"] = {m: [n for n, c in [("RBS>0", P.loc[m, "rbs"] > 0), ("L>=.95", P.loc[m, "L"] >= .95),
                                             ("F>=.50", P.loc[m, "F"] >= .5), ("Fp>=.50", P.loc[m, "Fp"] >= .5)] if c]
                          for m in P.index[t1]}
rest = P[~t1]
tier2 = {}
for col in ["L", "F", "Fp", "rbs", "K"]:
    mu, sd = rest[col].mean(), rest[col].std(ddof=1)
    for m in rest.index:
        z = (rest.loc[m, col] - mu) / sd
        if z >= 1.5:
            tier2.setdefault(m, []).append(f"{col} z={z:.2f}")
X["tier2_17707_rules"] = tier2
X["K_by_model"] = P.K.round(3).to_dict()
X["F_by_model"] = P.F.round(3).to_dict()

# item sensitivity under 17707 grouping (Tier 1 vs rest)
a, b = P.loc[~t1, "r"], P.loc[t1, "r"]
sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
X["r_tier1_vs_rest"] = dict(n_rest=len(a), mean_rest=a.mean(), sd_rest=a.std(), n_t1=len(b), mean_t1=b.mean(),
                            sd_t1=b.std(), d=(a.mean() - b.mean()) / sp,
                            welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue,
                            n_sig_rest=int(((P.loc[~t1, "r"] > 0) & (P.loc[~t1, "r_flag"].str[2:].astype(float) < .05)).sum()))
loo = []
for m in P.index[t1]:
    bb = b.drop(m)
    if len(bb) >= 2:
        loo.append((m, stats.ttest_ind(a, bb, equal_var=False).pvalue))
X["r_tier1_loo_welch_p"] = dict(loo)

# incremental prediction (17707 section 3.6), models with L defined
q = P.dropna(subset=["L", "withdraw_delta"])
for dv in ["withdraw_delta", "r"]:
    m0 = sm.OLS(q[dv], sm.add_constant(q[["accuracy"]])).fit()
    m1 = sm.OLS(q[dv], sm.add_constant(q[["accuracy", "L"]])).fit()
    f, p, _ = m1.compare_f_test(m0)
    X[f"incremental_{dv}"] = dict(n=len(q), r2_acc=m0.rsquared, r2_acc_L=m1.rsquared, dR2=m1.rsquared - m0.rsquared,
                                  F=f, p=p)

# retrospective (5 tracks) vs prospective L, corrected
rp = []
for m, g in S.groupby("model"):
    for lbl, sub in [("retro", g[g.track != "prospective"]), ("prosp", g[g.track == "prospective"])]:
        e = sub[sub.correct == 0]
        rp.append(dict(model=m, fmt=lbl, errors=len(e), L=e.keep.mean() if len(e) else np.nan))
RP = pd.DataFrame(rp).pivot(index="model", columns="fmt", values=["L", "errors"])
X["retro_prosp"] = RP.round(3).to_dict()
X["prosp_L1_models_corrected"] = sorted(RP.index[(RP[("L", "prosp")] == 1.0)])
X["prosp_errors_total_corrected"] = int(RP[("errors", "prosp")].sum())
# same, all rows (as published)
rp2 = []
for m, g in D[D.track == "prospective"].groupby("model"):
    e = g[g.correct == 0]
    rp2.append((m, len(e), e.keep.mean() if len(e) else np.nan))
X["prosp_all_rows"] = {m: dict(errors=n, L=l) for m, n, l in rp2}

# family summaries (corrected)
X["family_gemini_L"] = P.loc[["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Flash", "Gemini 3.1 Pro"], "L"].agg(
    ["mean", "std", "min", "max"]).to_dict()
X["family_gemini_K"] = P.loc[["Gemini 2.5 Flash", "Gemini 2.5 Pro", "Gemini 3 Flash", "Gemini 3.1 Pro"], "K"].agg(
    ["min", "max"]).to_dict()
X["openai"] = P.loc[["GPT-5.4 nano", "GPT-5.4 mini", "GPT-5.4"], ["accuracy", "L", "withdraw_delta"]].round(3).to_dict()
X["anthropic"] = P.loc[["Opus 4.6", "Sonnet 4.6", "Claude Haiku 4.5"], ["accuracy", "L", "withdraw_delta"]].round(3).to_dict()
X["pairs"] = P.loc[["DeepSeek V3.2", "DeepSeek-R1", "Qwen 80B Inst", "Qwen 80B Think"],
                   ["Fp", "K", "L", "withdraw_delta", "wd_bet_rate_correct", "r"]].round(3).to_dict()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


(OUT / "numbers_17707.json").write_text(json.dumps(clean(X), indent=1))
print(json.dumps(clean({k: v for k, v in X.items() if k != "retro_prosp"}), indent=1))
