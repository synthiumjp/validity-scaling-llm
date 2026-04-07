"""
Validity Scaling for LLM Metacognitive Self-Report
Main analysis pipeline

Computes six validity indices (L, K, F, Fp, RBS, TRIN) from 
dual-probe metacognitive data (KEEP/WITHDRAW + BET/NO BET).

Author: Jon-Paul Cacioli
Date: April 2026
"""

import pandas as pd
import numpy as np
import os
import glob
from scipy import stats


# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

def load_battery(base_path="data/csvs"):
    """Load all 120 CSVs from the Classical Minds battery."""
    
    tracks_map = {
        "T1": "Overhypothesis",
        "T2": "Meta Cog",
        "T3": "Social Cognition",
        "T4": "Attention",
        "T5": "Executive",
        "T6": "prospective"
    }
    
    all_rows = []
    for track_id, folder in tracks_map.items():
        path = os.path.join(base_path, folder)
        for f in sorted(glob.glob(os.path.join(path, "*.csv"))):
            df = pd.read_csv(f)
            model_name = _clean_model_name(os.path.basename(f))
            df["model"] = model_name
            df["track"] = track_id
            
            # Standardise correct column
            if "is_correct" in df.columns and "correct" not in df.columns:
                df["correct"] = df["is_correct"]
            df["correct"] = (df["correct"].astype(str).str.strip().str.lower()
                            .map({"true": True, "false": False}))
            
            # Create unique item ID
            if "item_id" in df.columns:
                df["item_uid"] = track_id + "_" + df["item_id"].astype(str)
            elif "item" in df.columns:
                df["item_uid"] = track_id + "_" + df["item"].astype(str)
            else:
                df["item_uid"] = track_id + "_" + df.index.astype(str)
            
            all_rows.append(df)
    
    data = pd.concat(all_rows, ignore_index=True)
    return data


def _clean_model_name(filename):
    """Clean model name from CSV filename."""
    mn = filename.replace(".csv", "")
    for suffix in [" attention", " metacog", " exec", " overhyp", " overhypothesis",
                   " social cog", " Social Cog", " Social cog", " soc cog",
                   "prospective_regulation_results", "exec", "metacog", "cog",
                   " Exec", "Exec"]:
        mn = mn.replace(suffix, "")
    mn = mn.strip()
    
    name_map = {
        "Opus 4.6": "Claude Opus 4.6",
        "haiku": "Claude Haiku 4.5",
        "R1": "DeepSeek-R1",
        "GLM 5": "GLM-5",
        "GLM-5": "GLM-5"
    }
    if mn in name_map:
        mn = name_map[mn]
    if "Gemini 3.1 Pro" in mn: mn = "Gemini 3.1 Pro"
    if "Gemini Flash 3" in mn or "Gemini 3 Flash" in mn: mn = "Gemini 3 Flash"
    if mn.startswith("GPT-5.4 mini"): mn = "GPT-5.4 mini"
    if mn.startswith("GLM 5") or mn == "GLM 5": mn = "GLM-5"
    if mn.startswith("Qwen 3 235B"): mn = "Qwen 235B"
    if mn.startswith("Qwen 3 Coder"): mn = "Qwen Coder 480B"
    if mn.startswith("Qwen 3 Next 80B Instruct"): mn = "Qwen 80B Instruct"
    if mn.startswith("Qwen 3 Next 80B Thinking"): mn = "Qwen 80B Think"
    if "R1" in mn and "DeepSeek" not in mn: mn = "DeepSeek-R1"
    
    return mn


# ============================================================
# ITEM-LEVEL NORMS
# ============================================================

def compute_item_norms(data):
    """Compute item-level normative KEEP rates across all models."""
    norms = data.groupby("item_uid").agg(
        norm_keep_rate=("keep_withdraw", lambda x: (x == "KEEP").mean()),
        norm_accuracy=("correct", "mean"),
        track=("track", "first"),
        n_models=("model", "nunique")
    ).reset_index()
    return norms


# ============================================================
# VALIDITY INDEX COMPUTATION
# ============================================================

def compute_validity_indices(data, item_norms=None):
    """
    Compute six validity indices for each model.
    
    Indices:
        L:    P(KEEP | incorrect) — under-reporting / blanket confidence
        K:    P(BET | incorrect) — unjustified strong confidence
        F:    P(WITHDRAW | item norm-KEEP >= 85%) — over-reporting
        Fp:   P(WITHDRAW | correct) — withdrawing correct answers
        RBS:  P(WD|correct) - P(WD|incorrect) — inverted monitoring
        TRIN: max(n_KEEP, n_WD) / n_total — fixed responding
    """
    if item_norms is None:
        item_norms = compute_item_norms(data)
    
    # Merge norms into data
    if "norm_keep_rate" not in data.columns:
        data = data.merge(
            item_norms[["item_uid", "norm_keep_rate"]],
            on="item_uid", how="left"
        )
    
    models = sorted(data.model.unique())
    results = []
    
    for model in models:
        md = data[data.model == model]
        correct = md[md.correct == True]
        incorrect = md[md.correct == False]
        
        # TRIN: Fixed responding
        n_keep = (md.keep_withdraw == "KEEP").sum()
        n_total = len(md)
        trin = max(n_keep, n_total - n_keep) / n_total
        
        # F: WITHDRAW on consensus-KEEP items (norm >= 85%)
        consensus_keep = md[md.norm_keep_rate >= 0.85]
        F = ((consensus_keep.keep_withdraw == "WITHDRAW").mean()
             if len(consensus_keep) > 0 else 0)
        
        # Fp: WITHDRAW on correct items
        Fp = ((correct.keep_withdraw == "WITHDRAW").mean()
              if len(correct) > 0 else 0)
        
        # RBS: P(WD|correct) - P(WD|incorrect)
        wd_correct = ((correct.keep_withdraw == "WITHDRAW").mean()
                      if len(correct) > 0 else 0)
        wd_incorrect = ((incorrect.keep_withdraw == "WITHDRAW").mean()
                        if len(incorrect) > 0 else 0)
        rbs = wd_correct - wd_incorrect
        withdraw_delta = wd_incorrect - wd_correct
        
        # L: P(KEEP | incorrect)
        L = ((incorrect.keep_withdraw == "KEEP").mean()
             if len(incorrect) > 0 else np.nan)
        
        # K: P(BET | incorrect)
        K = ((incorrect.bet_nobet == "BET").mean()
             if len(incorrect) > 0 else np.nan)
        
        # Item sensitivity: r(KEEP, correct)
        keep_vec = (md.keep_withdraw == "KEEP").astype(int).values
        correct_vec = md.correct.astype(int).values
        if keep_vec.std() > 0 and correct_vec.std() > 0:
            item_sensitivity, is_p = stats.pointbiserialr(correct_vec, keep_vec)
        else:
            item_sensitivity, is_p = 0.0, 1.0
        
        # Contradiction rate: P(WITHDRAW and BET)
        contradiction = ((md.keep_withdraw == "WITHDRAW") & 
                         (md.bet_nobet == "BET")).mean()
        
        # BET delta
        if len(correct) > 0 and len(incorrect) > 0:
            bet_delta = ((correct.bet_nobet == "BET").mean() - 
                        (incorrect.bet_nobet == "BET").mean())
        else:
            bet_delta = np.nan
        
        results.append({
            "model": model,
            "accuracy": md.correct.mean(),
            "n_correct": len(correct),
            "n_incorrect": len(incorrect),
            "TRIN": trin,
            "F": F,
            "Fp": Fp,
            "RBS": rbs,
            "L": L,
            "K": K,
            "withdraw_delta": withdraw_delta,
            "bet_delta": bet_delta,
            "item_sensitivity": item_sensitivity,
            "item_sensitivity_p": is_p,
            "contradiction_rate": contradiction,
        })
    
    return pd.DataFrame(results)


# ============================================================
# TIERED CLASSIFICATION
# ============================================================

def classify_tier1(row):
    """
    Tier 1: construct-level invalidity.
    Theory-driven thresholds, validated against synthetic policies.
    """
    flags = []
    if row.RBS > 0:
        flags.append("RBS+(inverted)")
    if row.L >= 0.95:
        flags.append("L>=.95(blanket)")
    if row.F >= 0.50:
        flags.append("F>=.50(extreme)")
    if row.Fp >= 0.50:
        flags.append("Fp>=.50(extreme)")
    return flags


def classify_tiered(indices_df):
    """Apply tiered classification to all models."""
    df = indices_df.copy()
    
    # Tier 1
    df["tier1_flags"] = df.apply(classify_tier1, axis=1)
    df["tier1_invalid"] = df.tier1_flags.apply(len) > 0
    
    # Tier 2: compute norms on Tier 1 valid models only
    valid_df = df[~df.tier1_invalid]
    tier2_thresholds = {}
    for scale in ["L", "K", "F", "Fp", "TRIN"]:
        vals = valid_df[scale].dropna()
        m, sd = vals.mean(), vals.std()
        tier2_thresholds[scale] = {
            "elevated": m + 1.5 * sd,
            "marked": m + 2.0 * sd
        }
    
    def classify_tier2(row):
        if row.tier1_invalid:
            return []
        flags = []
        for scale in ["L", "K", "F", "Fp"]:
            if row[scale] >= tier2_thresholds[scale]["marked"]:
                flags.append(f"{scale}>=marked")
            elif row[scale] >= tier2_thresholds[scale]["elevated"]:
                flags.append(f"{scale}>=elevated")
        return flags
    
    df["tier2_flags"] = df.apply(classify_tier2, axis=1)
    
    # Final tier assignment
    def assign_tier(row):
        if row.tier1_invalid:
            return "Tier 1"
        elif len(row.tier2_flags) > 0:
            if any("marked" in f for f in row.tier2_flags):
                return "Tier 2 (Caution)"
            return "Tier 2 (Elevated)"
        return "Valid"
    
    df["tier"] = df.apply(assign_tier, axis=1)
    
    return df, tier2_thresholds


# ============================================================
# PSYCHOMETRIC VALIDATION
# ============================================================

def cronbach_alpha(items_df):
    """Compute Cronbach's alpha from a subjects x items matrix."""
    k = items_df.shape[1]
    if k < 2:
        return np.nan
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def cohens_d(group1, group2):
    """Compute Cohen's d with pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = np.sqrt(
        ((n1 - 1) * group1.std(ddof=1)**2 + 
         (n2 - 1) * group2.std(ddof=1)**2) / (n1 + n2 - 2)
    )
    if pooled == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading battery data...")
    data = load_battery()
    print(f"  {len(data)} rows, {data.model.nunique()} models")
    
    print("\nComputing item norms...")
    item_norms = compute_item_norms(data)
    print(f"  {len(item_norms)} items")
    print(f"  Consensus KEEP (>=85%): {(item_norms.norm_keep_rate >= 0.85).sum()}")
    
    print("\nComputing validity indices...")
    indices = compute_validity_indices(data, item_norms)
    
    print("\nApplying tiered classification...")
    classified, t2_thresholds = classify_tiered(indices)
    
    # Print results
    print("\n" + "=" * 100)
    print("TIERED VALIDITY CLASSIFICATION")
    print("=" * 100)
    print(f"\n{'Model':25s} {'Tier':>12s} {'Acc':>5s} {'dw':>7s} {'L':>5s} "
          f"{'K':>5s} {'F':>5s} {'Fp':>5s} {'IS':>6s}")
    print("-" * 85)
    for _, r in classified.sort_values("accuracy", ascending=False).iterrows():
        print(f"{r.model:25s} {r.tier:>12s} {r.accuracy:5.3f} "
              f"{r.withdraw_delta:+7.3f} {r.L:5.3f} {r.K:5.3f} "
              f"{r.F:5.3f} {r.Fp:5.3f} {r.item_sensitivity:+6.3f}")
    
    # Item sensitivity comparison
    valid = classified[classified.tier == "Valid"]
    invalid = classified[classified.tier == "Tier 1"]
    
    print(f"\n{'='*60}")
    print("ITEM SENSITIVITY: VALID vs INVALID")
    print(f"{'='*60}")
    print(f"  Valid (n={len(valid)}):   M={valid.item_sensitivity.mean():.3f}, "
          f"SD={valid.item_sensitivity.std():.3f}")
    print(f"  Invalid (n={len(invalid)}): M={invalid.item_sensitivity.mean():.3f}, "
          f"SD={invalid.item_sensitivity.std():.3f}")
    d = cohens_d(valid.item_sensitivity, invalid.item_sensitivity)
    t, p = stats.ttest_ind(valid.item_sensitivity, invalid.item_sensitivity)
    print(f"  Cohen's d = {d:+.2f}, t = {t:.2f}, p = {p:.4f}")
    
    # Convergent / discriminant
    print(f"\n{'='*60}")
    print("CONVERGENT / DISCRIMINANT VALIDITY")
    print(f"{'='*60}")
    for a, b, label in [
        ("L", "K", "L vs K (under-reporting)"),
        ("F", "Fp", "F vs Fp (over-reporting)"),
        ("L", "F", "L vs F (discriminant)")
    ]:
        r, p = stats.pearsonr(classified[a], classified[b])
        print(f"  {label:35s} r = {r:.3f}, p = {p:.4f}")
    
    # Save
    classified.to_csv("analysis/validity_indices.csv", index=False)
    print("\nSaved to analysis/validity_indices.csv")
