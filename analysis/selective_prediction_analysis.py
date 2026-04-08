"""
Project E: Selective Prediction as Deployment-Facing Criterion
Full analysis pipeline using validity_screen.py for tier assignment.

Run from the repository root:
    python analysis/project_e_analysis_v3.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, f_oneway, pointbiserialr
import sys
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "csvs"
sys.path.insert(0, str(REPO_ROOT / "screen"))
from validity_screen import screen

TRACK_MAP = {"Attention":"T1","Executive":"T2","Meta Cog":"T3",
             "Overhypothesis":"T4","Social Cognition":"T5","prospective":"T6"}

MODEL_PATTERNS = [
    ("GPT-5.4 nano","GPT-5.4 nano"),("GPT-5.4 mini","GPT-5.4 mini"),("GPT-5.4","GPT-5.4"),
    ("Claude Opus 4.6","Opus 4.6"),("Opus 4.6","Opus 4.6"),
    ("Claude Sonnet 4.6","Sonnet 4.6"),("Sonnet 4.6","Sonnet 4.6"),
    ("Claude Haiku 4.5","Claude Haiku 4.5"),("haiku","Claude Haiku 4.5"),
    ("DeepSeek-R1","DeepSeek-R1"),("R1","DeepSeek-R1"),
    ("DeepSeek V3.2","DeepSeek V3.2"),
    ("Gemini 3.1 Pro","Gemini 3.1 Pro"),("Gemini 2.5 Pro","Gemini 2.5 Pro"),
    ("Gemini 2.5 Flash","Gemini 2.5 Flash"),
    ("Gemini 3 Flash Preview","Gemini 3 Flash"),("Gemini Flash 3","Gemini 3 Flash"),
    ("Gemini 3 Flash","Gemini 3 Flash"),
    ("Gemma 3 1B","Gemma 3 1B"),("Gemma 3 12B","Gemma 3 12B"),("Gemma 3 27B","Gemma 3 27B"),
    ("GLM-5","GLM-5"),("GLM 5","GLM-5"),
    ("Qwen 3 235B","Qwen 235B"),("Qwen 3 Coder 480B","Qwen Coder 480B"),
    ("Qwen 3 Next 80B Thinking","Qwen 80B Think"),("Qwen 3 Next 80B Think","Qwen 80B Think"),
    ("Qwen 3 Next 80B Instruct","Qwen 80B Inst"),("Qwen 3 Next 80B Inst","Qwen 80B Inst"),
]

FAMILY_MAP = {
    "Claude Haiku 4.5":"Anthropic","Sonnet 4.6":"Anthropic","Opus 4.6":"Anthropic",
    "GPT-5.4":"OpenAI","GPT-5.4 mini":"OpenAI","GPT-5.4 nano":"OpenAI",
    "Gemini 2.5 Flash":"Google","Gemini 2.5 Pro":"Google","Gemini 3 Flash":"Google",
    "Gemini 3.1 Pro":"Google","Gemma 3 1B":"Google","Gemma 3 12B":"Google","Gemma 3 27B":"Google",
    "Qwen 235B":"Qwen","Qwen 80B Inst":"Qwen","Qwen 80B Think":"Qwen","Qwen Coder 480B":"Qwen",
    "DeepSeek V3.2":"DeepSeek","DeepSeek-R1":"DeepSeek","GLM-5":"Zhipu",
}

OVERRIDE_INVALID = {"Gemini 3.1 Pro", "Qwen 80B Think"}

def match_model(stem):
    s = stem.lower()
    for pat, canon in MODEL_PATTERNS:
        if pat.lower() in s: return canon
    return None

def load_all():
    rows = []
    for td, tc in TRACK_MAP.items():
        track_path = DATA_DIR / td
        if not track_path.exists():
            print(f"  WARNING: track directory not found: {track_path}")
            continue
        for f in track_path.glob("*.csv"):
            df = pd.read_csv(f)
            cc = "correct" if "correct" in df.columns else "is_correct" if "is_correct" in df.columns else None
            if not cc: continue
            df["correct_bool"] = df[cc].apply(lambda x: str(x).strip().lower() in ["true","1","1.0"])
            df["confidence"] = (df["keep_withdraw"].str.strip().str.upper()=="KEEP").astype(int)
            df["bet"] = (df["bet_nobet"].str.strip().str.upper()=="BET").astype(int) if "bet_nobet" in df.columns else 0
            df["confidence_ordinal"] = df["confidence"]*2 + df["bet"]
            mn = match_model(f.stem)
            if not mn: continue
            df["model"], df["track"] = mn, tc
            rows.append(df[["model","track","correct_bool","confidence","bet","confidence_ordinal"]])
    return pd.concat(rows, ignore_index=True)

def compute_auroc(dm):
    y, s = dm["correct_bool"].astype(int).values, dm["confidence_ordinal"].values
    if len(np.unique(y))<2 or len(np.unique(s))<2: return np.nan
    return roc_auc_score(y, s)

def sel_acc(dm, cov):
    k = max(1, int(np.ceil(len(dm)*cov)))
    return dm.sort_values("confidence_ordinal", ascending=False, kind="mergesort").iloc[:k]["correct_bool"].mean()

def sel_gain(dm, cov):
    return sel_acc(dm, cov) - dm["correct_bool"].mean()


if __name__ == "__main__":
    # ===========================================================
    # 1. LOAD DATA AND COMPUTE TIERS VIA SCREEN
    # ===========================================================
    print("="*70)
    print("STEP 1: Loading data and running validity screen")
    print("="*70)
    print(f"  Data directory: {DATA_DIR}")
    df = load_all()
    print(f"  Loaded {len(df)} items, {df['model'].nunique()} models, {df['track'].nunique()} tracks")

    tier_map = {}
    screen_results = []
    for model_name in sorted(df["model"].unique()):
        dm = df[df["model"]==model_name]
        correct = dm["correct_bool"].values.astype(bool)
        confidence = dm["confidence"].values.astype(bool)
        
        sr = screen(correct, confidence, model_name=model_name,
                    benchmark_name="Classical Minds Battery",
                    elicitation_method="Binary probe (KEEP/WITHDRAW)",
                    confidence_format="Binary", probe_timing="Retrospective")
        
        if sr.tier == "Insufficient data" and model_name in OVERRIDE_INVALID:
            sr.tier = "Invalid"
            sr.flagging_reasons.append(f"Override: L={sr.L.value if sr.L else 'N/A'} from derivation study (cell d < 5)")
        elif sr.tier == "Insufficient data":
            r_val, p_val = pointbiserialr(confidence.astype(int), correct.astype(int))
            if r_val > 0 and p_val < 0.05:
                sr.tier = "Valid"
                sr.flagging_reasons.append(f"Override: r={r_val:.3f}, p={p_val:.3f}, positive and significant (cell d < 5)")
            else:
                sr.tier = "Indeterminate"
                sr.flagging_reasons.append(f"Override: insufficient data, r={r_val:.3f}, p={p_val:.3f}")
        
        tier_map[model_name] = sr.tier
        screen_results.append(sr)
        
        L_val = f"{sr.L.value:.3f}" if sr.L else "N/A"
        Fp_val = f"{sr.Fp.value:.3f}" if sr.Fp else "N/A"
        r_val = f"{sr.r_conf_correct.value:+.3f}" if sr.r_conf_correct else "N/A"
        print(f"  {model_name:25s}  tier={sr.tier:15s}  L={L_val}  Fp={Fp_val}  r={r_val}  {'; '.join(sr.flagging_reasons[:1])}")

    df["tier"] = df["model"].map(tier_map)
    df["family"] = df["model"].map(FAMILY_MAP)

    tier_counts = {t: sum(1 for v in tier_map.values() if v==t) for t in ["Valid","Indeterminate","Invalid"]}
    print(f"\n  Tier counts: {tier_counts}")

    # ===========================================================
    # 2. MAIN RESULTS
    # ===========================================================
    print(f"\n{'='*70}")
    print("STEP 2: Selective prediction metrics")
    print("="*70)

    results = []
    for m in sorted(df["model"].unique()):
        dm = df[df["model"]==m]
        t = tier_map[m]
        fam = FAMILY_MAP[m]
        a = compute_auroc(dm)
        ba = dm["correct_bool"].mean()
        results.append({"model":m,"tier":t,"family":fam,"n":len(dm),"baseline":ba,
                         "auroc":a,
                         "gain_90":sel_gain(dm,.9),"gain_80":sel_gain(dm,.8),
                         "gain_70":sel_gain(dm,.7),"gain_60":sel_gain(dm,.6),
                         "gain_50":sel_gain(dm,.5),"gain_30":sel_gain(dm,.3),
                         "gain_20":sel_gain(dm,.2)})

    rdf = pd.DataFrame(results)

    print(f"\n  {'Model':25s} {'Tier':15s} {'Base':>6s} {'AUROC':>6s} {'G@80':>6s} {'G@70':>6s} {'G@50':>6s}")
    print("  " + "-"*75)
    for _, r in rdf.sort_values(["tier","auroc"], ascending=[True,False]).iterrows():
        print(f"  {r['model']:25s} {r['tier']:15s} {r['baseline']:6.3f} {r['auroc']:6.3f} {r['gain_80']:+6.3f} {r['gain_70']:+6.3f} {r['gain_50']:+6.3f}")

    # ===========================================================
    # 3. BOOTSTRAP CIs ON TIER MEANS
    # ===========================================================
    print(f"\n{'='*70}")
    print("STEP 3: Bootstrap CIs on tier means (AUROC)")
    print("="*70)

    np.random.seed(42)
    n_boot = 10000
    for tier_name in ["Valid","Indeterminate","Invalid"]:
        vals = rdf[rdf["tier"]==tier_name]["auroc"].values
        n = len(vals)
        boot_means = np.array([np.mean(np.random.choice(vals, n, replace=True)) for _ in range(n_boot)])
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"  {tier_name:15s} (n={n}): mean={vals.mean():.3f}, 95% boot CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    boot_mono = 0
    for _ in range(n_boot):
        v_boot = np.mean(np.random.choice(rdf[rdf["tier"]=="Valid"]["auroc"].values, 
                         len(rdf[rdf["tier"]=="Valid"]), replace=True))
        i_boot = np.mean(np.random.choice(rdf[rdf["tier"]=="Indeterminate"]["auroc"].values,
                         len(rdf[rdf["tier"]=="Indeterminate"]), replace=True))
        inv_boot = np.mean(np.random.choice(rdf[rdf["tier"]=="Invalid"]["auroc"].values,
                           len(rdf[rdf["tier"]=="Invalid"]), replace=True))
        if inv_boot < i_boot < v_boot:
            boot_mono += 1
    print(f"\n  P(monotonic ordering) = {boot_mono/n_boot:.3f} (bootstrap, {n_boot} samples)")

    # ===========================================================
    # 4. SPLIT-HALF CROSS-VALIDATION
    # ===========================================================
    print(f"\n{'='*70}")
    print("STEP 4: Split-half cross-validation")
    print("="*70)

    np.random.seed(123)
    n_splits = 1000
    split_d_values = []
    split_auroc_diffs = []

    for _ in range(n_splits):
        half_results = []
        for m in df["model"].unique():
            dm = df[df["model"]==m].copy()
            idx = np.random.permutation(len(dm))
            half1 = dm.iloc[idx[:len(dm)//2]]
            half2 = dm.iloc[idx[len(dm)//2:]]
            
            correct1 = half1["correct_bool"].values.astype(bool)
            conf1 = half1["confidence"].values.astype(bool)
            sr1 = screen(correct1, conf1, model_name=m)
            tier1 = sr1.tier
            if tier1 == "Insufficient data":
                if m in OVERRIDE_INVALID:
                    tier1 = "Invalid"
                else:
                    r_v, p_v = pointbiserialr(conf1.astype(int), correct1.astype(int))
                    tier1 = "Valid" if (r_v > 0 and p_v < 0.05) else "Indeterminate"
            
            y2 = half2["correct_bool"].astype(int).values
            s2 = half2["confidence_ordinal"].values
            if len(np.unique(y2)) >= 2 and len(np.unique(s2)) >= 2:
                auroc2 = roc_auc_score(y2, s2)
            else:
                auroc2 = np.nan
            
            half_results.append({"model":m, "tier_train":tier1, "auroc_test":auroc2})
        
        hdf = pd.DataFrame(half_results).dropna()
        v_auroc = hdf[hdf["tier_train"]=="Valid"]["auroc_test"].values
        inv_auroc = hdf[hdf["tier_train"]=="Invalid"]["auroc_test"].values
        
        if len(v_auroc) >= 2 and len(inv_auroc) >= 2:
            diff = v_auroc.mean() - inv_auroc.mean()
            split_auroc_diffs.append(diff)
            ps = np.sqrt(((len(v_auroc)-1)*v_auroc.std()**2 + (len(inv_auroc)-1)*inv_auroc.std()**2) /
                         (len(v_auroc)+len(inv_auroc)-2))
            if ps > 0:
                split_d_values.append(diff/ps)

    split_d = np.array(split_d_values)
    split_diffs = np.array(split_auroc_diffs)
    print(f"  Split-half cross-validation ({n_splits} splits):")
    print(f"  Mean d (Valid vs Invalid): {split_d.mean():.2f} (SD={split_d.std():.2f})")
    print(f"  Median d: {np.median(split_d):.2f}")
    print(f"  95% CI on d: [{np.percentile(split_d, 2.5):.2f}, {np.percentile(split_d, 97.5):.2f}]")
    print(f"  P(d > 0): {(split_d > 0).mean():.4f}")
    print(f"  Mean AUROC diff: {split_diffs.mean():.3f} [{np.percentile(split_diffs, 2.5):.3f}, {np.percentile(split_diffs, 97.5):.3f}]")

    # ===========================================================
    # 5. FAMILY-LEVEL CLUSTERING CHECK
    # ===========================================================
    print(f"\n{'='*70}")
    print("STEP 5: Family-level clustering check")
    print("="*70)

    family_reps = []
    for fam in rdf["family"].unique():
        fam_models = rdf[rdf["family"]==fam]
        med = fam_models["auroc"].median()
        closest = fam_models.iloc[(fam_models["auroc"]-med).abs().argsort()[:1]]
        family_reps.append(closest.iloc[0])

    fam_df = pd.DataFrame(family_reps)
    print(f"  Family representatives (n={len(fam_df)}):")
    for _, r in fam_df.iterrows():
        print(f"    {r['family']:12s} -> {r['model']:25s}  tier={r['tier']:15s}  AUROC={r['auroc']:.3f}")

    v_fam = fam_df[fam_df["tier"]=="Valid"]["auroc"].values.astype(float)
    inv_fam = fam_df[fam_df["tier"]=="Invalid"]["auroc"].values.astype(float)
    if len(v_fam) >= 2 and len(inv_fam) >= 1:
        print(f"\n  Family-level Valid mean: {v_fam.mean():.3f}")
        print(f"  Family-level Invalid mean: {inv_fam.mean():.3f}")
        if len(inv_fam) >= 2:
            u, p = mannwhitneyu(v_fam, inv_fam, alternative="greater")
            print(f"  Mann-Whitney: U={u:.0f}, p={p:.4f}")

    # ===========================================================
    # 6. PER-TRACK VALIDITY PREDICTING PER-TRACK SELECTIVE PERF
    # ===========================================================
    print(f"\n{'='*70}")
    print("STEP 6: Per-track validity predicting per-track AUROC")
    print("="*70)

    track_results = []
    for m in sorted(df["model"].unique()):
        for t in sorted(df["track"].unique()):
            dm = df[(df["model"]==m)&(df["track"]==t)]
            if len(dm) < 10: continue
            
            correct = dm["correct_bool"].values.astype(bool)
            confidence = dm["confidence"].values.astype(bool)
            
            if len(np.unique(confidence)) >= 2 and len(np.unique(correct)) >= 2:
                r_val, p_val = pointbiserialr(confidence.astype(int), correct.astype(int))
            else:
                r_val, p_val = 0, 1
            
            auroc = compute_auroc(dm)
            
            track_results.append({"model":m,"track":t,"tier":tier_map[m],
                                  "r_track":r_val,"auroc_track":auroc if not np.isnan(auroc) else None,
                                  "n":len(dm)})

    tdf = pd.DataFrame(track_results).dropna(subset=["auroc_track"])
    rho, p = spearmanr(tdf["r_track"], tdf["auroc_track"])
    print(f"  Per-track r(conf,correct) vs per-track AUROC:")
    print(f"    Spearman rho = {rho:.3f}, p = {p:.6f}, n_obs = {len(tdf)}")

    tv = tdf[tdf["tier"]=="Valid"]
    rho_v, p_v = spearmanr(tv["r_track"], tv["auroc_track"])
    print(f"  Valid models only: rho = {rho_v:.3f}, p = {p_v:.6f}, n_obs = {len(tv)}")

    print("\nDone.")
