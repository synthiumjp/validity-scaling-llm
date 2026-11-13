"""Full rescore of the 20-model battery with validity_screen v0.2.

Run:  [VS_DROP_TRACKS=prospective] python rescore.py <battery csv dir> <outdir> (see run_all.sh)
Writes per_model.csv, per_track.csv and numbers.json (every number used in the
corrected preprints comes from numbers.json).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validity_screen import screen, is_nonanswer, wilson_ci  # noqa: E402

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
NUM = {}

MODEL_PATTERNS = [
    ("GPT-5.4 nano", "GPT-5.4 nano"), ("GPT-5.4 mini", "GPT-5.4 mini"), ("GPT-5.4", "GPT-5.4"),
    ("Claude Opus 4.6", "Opus 4.6"), ("Claude Sonnet 4.6", "Sonnet 4.6"),
    ("Claude Haiku 4.5", "Claude Haiku 4.5"), ("DeepSeek-R1", "DeepSeek-R1"),
    ("DeepSeek V3.2", "DeepSeek V3.2"), ("Gemini 3.1 Pro", "Gemini 3.1 Pro"),
    ("Gemini 2.5 Pro", "Gemini 2.5 Pro"), ("Gemini 2.5 Flash", "Gemini 2.5 Flash"),
    ("Gemini 3 Flash", "Gemini 3 Flash"), ("Gemma 3 1B", "Gemma 3 1B"),
    ("Gemma 3 12B", "Gemma 3 12B"), ("Gemma 3 27B", "Gemma 3 27B"), ("GLM-5", "GLM-5"), ("GLM 5", "GLM-5"), ("haiku", "Claude Haiku 4.5"), ("Opus 4.6", "Opus 4.6"), ("Sonnet 4.6", "Sonnet 4.6"), ("Gemini Flash 3", "Gemini 3 Flash"),
    ("Qwen 3 235B", "Qwen 235B"), ("Qwen 3 Coder 480B", "Qwen Coder 480B"),
    ("Qwen 3 Next 80B Thinking", "Qwen 80B Think"), ("Qwen 3 Next 80B Think", "Qwen 80B Think"), ("Qwen 3 Next 80B Instruct", "Qwen 80B Inst"), ("Qwen 3 Next 80B Inst", "Qwen 80B Inst"), ("Gemini 3 Flash Preview", "Gemini 3 Flash"), ("R1", "DeepSeek-R1"),
]
TRACKS = [("attention", "attention"), ("exec", "executive"), ("metacog", "metacognition"),
          ("overhyp", "learning"), ("socialcog", "social"), ("soccog", "social"), ("prospective", "prospective")]
FAMILY = {"Claude Haiku 4.5": "Anthropic", "Opus 4.6": "Anthropic", "Sonnet 4.6": "Anthropic",
          "GPT-5.4": "OpenAI", "GPT-5.4 mini": "OpenAI", "GPT-5.4 nano": "OpenAI",
          "Gemini 2.5 Flash": "Gemini", "Gemini 2.5 Pro": "Gemini", "Gemini 3 Flash": "Gemini",
          "Gemini 3.1 Pro": "Gemini", "Gemma 3 1B": "Gemma", "Gemma 3 12B": "Gemma",
          "Gemma 3 27B": "Gemma", "Qwen 235B": "Qwen", "Qwen 80B Inst": "Qwen",
          "Qwen 80B Think": "Qwen", "Qwen Coder 480B": "Qwen", "DeepSeek V3.2": "DeepSeek",
          "DeepSeek-R1": "DeepSeek", "GLM-5": "Zhipu"}
PUBLISHED_INVALID = ["DeepSeek-R1", "Gemini 3.1 Pro", "Qwen 80B Think"]


def model_of(stem):
    s = stem.lower()
    for pat, canon in MODEL_PATTERNS:
        if pat.lower() in s:
            return canon
    raise ValueError(stem)


def track_of(stem):
    s = stem.lower().replace(" ", "")
    for k, t in TRACKS:
        if k in s:
            return t
    raise ValueError(stem)


# ---------------------------------------------------------------- load
rows = []
for f in sorted(DATA.glob("*.csv")):
    df = pd.read_csv(f)
    cc = "correct" if "correct" in df.columns else "is_correct"
    o = pd.DataFrame({
        "model": model_of(f.stem), "track": track_of(f.stem), "pos": np.arange(len(df)),
        "correct": df[cc].map(lambda x: str(x).strip().lower() in ["true", "1", "1.0"]).astype(int),
        "keep": (df["keep_withdraw"].astype(str).str.strip().str.upper() == "KEEP").astype(int),
        "bet": (df["bet_nobet"].astype(str).str.strip().str.upper() == "BET").astype(int),
    })
    if "answer" in df.columns and "correct_answer" in df.columns:
        o["answered"] = [int(not is_nonanswer(a, ca)) for a, ca in zip(df["answer"], df["correct_answer"])]
    else:
        o["answered"] = 1
    o["path"] = df["path_choice"].astype(str) if "path_choice" in df.columns else "NA"
    rows.append(o)
D = pd.concat(rows, ignore_index=True)
# Optional track exclusion, e.g. VS_DROP_TRACKS=prospective (free-text track whose
# substring grader is unreliable in both directions; see paper Sec. 3).
import os  # noqa: E402
DROP = [t for t in os.environ.get("VS_DROP_TRACKS", "").split(",") if t]
D = D[~D.track.isin(DROP)].reset_index(drop=True)
NUM_DROP = DROP
D["ordinal"] = D.keep * 2 + D.bet
D["item"] = D.track + ":" + D.pos.astype(str)
NUM["dropped_tracks"] = NUM_DROP
NUM["n_models"] = int(D.model.nunique())
NUM["n_rows"] = int(len(D))
NUM["n_nonanswer"] = int((D.answered == 0).sum())
NUM["nonanswer_by_model"] = D[D.answered == 0].groupby("model").size().to_dict()
NUM["nonanswer_decline"] = int(((D.answered == 0) & (D.path == "DECLINE")).sum())
NUM["nonanswer_decline_conf"] = D[(D.path == "DECLINE")].groupby(["keep", "bet"]).size().rename(
    lambda t: f"keep{t[0]}_bet{t[1]}").to_dict() if False else \
    {f"keep{k}_bet{b}": int(v) for (k, b), v in D[D.path == "DECLINE"].groupby(["keep", "bet"]).size().items()}
S = D[D.answered == 1].copy()

# ---------------------------------------------------------------- per model


def tie_avg_sel_acc(conf, cor, coverage):
    """Expected accuracy among the top `coverage` fraction, ties broken uniformly at random."""
    n = len(conf)
    k = int(np.ceil(n * coverage))
    order = sorted(set(conf), reverse=True)
    taken, hits = 0, 0.0
    for lvl in order:
        m = conf == lvl
        cnt = m.sum()
        if taken + cnt <= k:
            hits += cor[m].sum()
            taken += cnt
        else:
            need = k - taken
            hits += need * cor[m].mean()
            taken = k
            break
    return hits / k


per = []
for m, g in S.groupby("model"):
    full = D[D.model == m]
    r = screen(full.correct.values, full.keep.values, full.answered.values, model_name=m)
    row = r.to_dict()
    row["family"] = FAMILY[m]
    row["auroc_ordinal"] = roc_auc_score(g.correct, g.ordinal)
    row["K"] = g.bet[g.correct == 0].mean() if (g.correct == 0).sum() else np.nan
    row["wd_bet_rate_all"] = ((g.keep == 0) & (g.bet == 1)).mean()
    row["wd_bet_rate_correct"] = ((g.keep == 0) & (g.bet == 1))[g.correct == 1].mean()
    row["withdraw_delta"] = (1 - g.keep[g.correct == 0].mean()) - (1 - g.keep[g.correct == 1].mean()) \
        if (g.correct == 0).sum() else np.nan
    row["bet_delta"] = (1 - g.bet[g.correct == 0].mean()) - (1 - g.bet[g.correct == 1].mean()) \
        if (g.correct == 0).sum() else np.nan
    for cov in [1.0, 0.5, 0.3, 0.1]:
        row[f"selacc_{int(cov*100)}"] = tie_avg_sel_acc(g.ordinal.values, g.correct.values, cov)
    per.append(row)
P = pd.DataFrame(per).set_index("model_name")
P.index.name = "model"

# F: withdrawal on consensus-KEEP items (>= .85 of models KEEP), scored items only
cons = S.groupby("item").keep.mean()
cons_items = cons[cons >= 0.85].index
P["F"] = S[S.item.isin(cons_items)].groupby("model").keep.apply(lambda k: 1 - k.mean())
NUM["F_consensus_items"] = int(len(cons_items))

P.to_csv(OUT / "per_model.csv")

# ---------------------------------------------------------------- tiers
NUM["tiers"] = P.tier.value_counts().to_dict()
NUM["tier_members"] = {t: sorted(P.index[P.tier == t]) for t in P.tier.unique()}
NUM["trin_warning_n"] = int((P.trin >= 0.95).sum())
NUM["rbs_positive_n"] = int((P.rbs > 0).sum())
NUM["rbs_max"] = float(P.rbs.max())
NUM["models_lt_35_errors"] = int((P.n_incorrect < 35).sum())


def wilson_lb_cont(p, n, z=1.959963984540054):
    """Wilson lower bound with the observed proportion held at p (continuous n)."""
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / den


def errors_needed(Lval=1.0, guard=0.90):
    """Smallest error count n at which an observed L = Lval has Wilson lower bound > guard
    (L held fixed, n treated as continuous, so the answer is monotone in Lval)."""
    for n in range(1, 5000):
        if wilson_lb_cont(Lval, n) > guard:
            return n


for _L in [1.00, 0.97, 0.96, 0.95]:
    NUM[f"errors_needed_L{_L:.2f}"] = errors_needed(_L)

# ---------------------------------------------------------------- group stats


def hedges(x, y):
    x, y = np.asarray(x), np.asarray(y)
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    d = (x.mean() - y.mean()) / sp
    return d * (1 - 3 / (4 * (len(x) + len(y)) - 9)), d


def perm_p(x, y, nperm=20000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    obs = abs(x.mean() - y.mean())
    pool = np.r_[x, y]
    cnt = 0
    for _ in range(nperm):
        rng.shuffle(pool)
        cnt += abs(pool[:len(x)].mean() - pool[len(x):].mean()) >= obs - 1e-12
    return (cnt + 1) / (nperm + 1)


def compare(col, grp_a, grp_b, label):
    x, y = P.loc[grp_a, col].values, P.loc[grp_b, col].values
    g, d = hedges(x, y)
    w = stats.ttest_ind(x, y, equal_var=False)
    mw = stats.mannwhitneyu(x, y, alternative="two-sided")
    NUM[label] = dict(n_a=len(x), n_b=len(y), mean_a=x.mean(), sd_a=x.std(ddof=1) if len(x) > 1 else np.nan,
                      mean_b=y.mean(), sd_b=y.std(ddof=1), d=d, hedges_g=g, welch_t=w.statistic,
                      welch_p=w.pvalue, mw_U=mw.statistic, mw_p_two_sided=mw.pvalue, perm_p=perm_p(x, y))


valid = list(P.index[P.tier == "Valid"])
nonvalid = list(P.index[P.tier != "Valid"])
compare("r", valid, nonvalid, "r_valid_vs_nonvalid")
compare("auroc_ordinal", valid, nonvalid, "auroc_valid_vs_nonvalid")
others = [m for m in P.index if m not in PUBLISHED_INVALID]
compare("r", others, PUBLISHED_INVALID, "r_published3_corrected_data")
compare("auroc_ordinal", others, PUBLISHED_INVALID, "auroc_published3_corrected_data")
NUM["tier_means_auroc"] = P.groupby("tier").auroc_ordinal.agg(["mean", "std", "count"]).to_dict("index")
NUM["tier_means_r"] = P.groupby("tier").r.agg(["mean", "std", "count"]).to_dict("index")

# circularity: screening index vs criterion
ok = P.rbs.notna()
NUM["corr_rbs_auroc_ordinal"] = stats.pearsonr(P.rbs[ok], P.auroc_ordinal[ok])[0]
NUM["corr_r_auroc_ordinal"] = stats.pearsonr(P.r, P.auroc_ordinal)[0]
NUM["mean_abs_diff_binary_vs_ordinal_auroc"] = float((P.auroc_binary - P.auroc_ordinal).abs()[ok].mean())

# baseline screen: r not significantly > 0 (two-sided p >= .05 or r <= 0)
P["r_p"] = P.r_flag.str.replace("p=", "").astype(float)
NUM["baseline_r_ns"] = sorted(P.index[(P.r <= 0) | (P.r_p >= 0.05)])

# ---------------------------------------------------------------- R1 and pairs
r1 = P.loc["DeepSeek-R1"]
NUM["R1"] = {k: float(r1[k]) for k in ["n_scored", "n_abstained", "accuracy", "n_incorrect", "Fp", "Fp_lo",
                                      "L", "rbs", "r", "auroc_binary", "auroc_ordinal", "trin",
                                      "selacc_100", "selacc_50", "selacc_30", "selacc_10",
                                      "wd_bet_rate_all", "wd_bet_rate_correct"]}
NUM["R1"]["keep_rate"] = float(S[S.model == "DeepSeek-R1"].keep.mean())
NUM["V32"] = {"withdraw_on_correct": float(P.loc["DeepSeek V3.2", "Fp"]), "L": float(P.loc["DeepSeek V3.2", "L"]),
              "r": float(P.loc["DeepSeek V3.2", "r"])}
NUM["QwenInst"] = {k: float(P.loc["Qwen 80B Inst", k]) for k in ["L", "L_lo", "r", "Fp", "K"]}
NUM["QwenThink"] = {k: float(P.loc["Qwen 80B Think", k]) for k in ["L", "L_lo", "r", "Fp", "K", "n_incorrect"]}
NUM["Gem31Pro"] = {k: float(P.loc["Gemini 3.1 Pro", k]) for k in ["L", "L_lo", "r", "Fp", "n_incorrect"]}
NUM["wd_bet_correct_top"] = P.wd_bet_rate_correct.sort_values(ascending=False).head(4).round(4).to_dict()
NUM["sonnet_selacc"] = {k: float(P.loc["Sonnet 4.6", k]) for k in ["selacc_100", "selacc_30", "selacc_10"]}

# ---------------------------------------------------------------- psychometrics
pt = []
for (m, t), g in S.groupby(["model", "track"]):
    e = g[g.correct == 0]
    pt.append(dict(model=m, track=t, n=len(g), errors=len(e),
                   L=e.keep.mean() if len(e) else np.nan, K=e.bet.mean() if len(e) else np.nan,
                   r=stats.pearsonr(g.keep, g.correct)[0] if g.keep.std() > 0 and g.correct.std() > 0 else np.nan,
                   auroc=roc_auc_score(g.correct, g.ordinal) if g.correct.nunique() > 1 and g.ordinal.nunique() > 1 else np.nan))
PT = pd.DataFrame(pt)
PT.to_csv(OUT / "per_track.csv", index=False)
NUM["cells_zero_errors"] = int((PT.errors == 0).sum())
NUM["cells_lt5_errors"] = int((PT.errors < 5).sum())


def cronbach(X):
    X = X.dropna()
    k = X.shape[1]
    return dict(n=len(X), alpha=k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1)))


NUM["alpha_L_complete_case"] = cronbach(PT.pivot(index="model", columns="track", values="L"))
NUM["alpha_K_complete_case"] = cronbach(PT.pivot(index="model", columns="track", values="K"))
pc = PT.dropna(subset=["r", "auroc"])
NUM["track_r_vs_auroc"] = dict(n=len(pc), rho=stats.spearmanr(pc.r, pc.auroc)[0])
ok = P.L.notna() & P.F.notna()
for a_, b_ in [("L", "K"), ("F", "Fp"), ("L", "F"), ("L", "accuracy"), ("withdraw_delta", "bet_delta")]:
    q = P[[a_, b_]].dropna()
    NUM[f"corr_{a_}_{b_}"] = dict(n=len(q), r=stats.pearsonr(q[a_], q[b_])[0])
v = P.loc[valid, ["L", "auroc_ordinal"]].dropna()
NUM["within_valid_L_auroc"] = dict(n=len(v), rho=stats.spearmanr(v.L, v.auroc_ordinal)[0],
                                   p=stats.spearmanr(v.L, v.auroc_ordinal)[1])

# ---------------------------------------------------------------- split-half stability of the frozen tool
rng = np.random.default_rng(42)
agree = {m: 0 for m in P.index}
for _ in range(1000):
    for m, g in D.groupby("model"):
        idx = rng.permutation(len(g))[: len(g) // 2]
        h = g.iloc[idx]
        rr = screen(h.correct.values, h.keep.values, h.answered.values)
        agree[m] += rr.tier == P.loc[m, "tier"]
NUM["splithalf_tier_agreement"] = {m: agree[m] / 1000 for m in P.index}

# ---------------------------------------------------------------- write
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


(OUT / "numbers.json").write_text(json.dumps(clean(NUM), indent=1))
cols = ["family", "n_scored", "n_abstained", "accuracy", "n_incorrect", "L", "L_lo", "Fp", "Fp_lo", "rbs", "rbs_lo",
        "rbs_hi", "trin", "r", "auroc_binary", "auroc_ordinal", "tier"]
pd.set_option("display.width", 250)
print(P[cols].sort_values("auroc_ordinal").round(3).to_string())
print(json.dumps(clean({k: NUM[k] for k in ["tiers", "tier_members", "n_nonanswer", "nonanswer_by_model"]}), indent=1))
