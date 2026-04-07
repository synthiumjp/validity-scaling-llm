"""
Robustness analyses for validity scaling paper.
- Bootstrap CI on Cohen's d for item-sensitivity comparison
- Leave-one-out sensitivity analysis
- Threshold sensitivity sweep
- PCA factor structure

Author: Jon-Paul Cacioli
Date: April 2026
"""

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_cohens_d(group1, group2, n_boot=10000, seed=42):
    """Bootstrap CI on Cohen's d."""
    np.random.seed(seed)
    boot_ds = []
    for _ in range(n_boot):
        bv = np.random.choice(group1, size=len(group1), replace=True)
        bi = np.random.choice(group2, size=len(group2), replace=True)
        n1, n2 = len(bv), len(bi)
        pooled = np.sqrt(
            ((n1-1)*bv.std(ddof=1)**2 + (n2-1)*bi.std(ddof=1)**2) / (n1+n2-2)
        )
        if pooled > 0:
            boot_ds.append((bv.mean() - bi.mean()) / pooled)
    boot_ds = np.array(boot_ds)
    return {
        "mean": np.mean(boot_ds),
        "ci_lower": np.percentile(boot_ds, 2.5),
        "ci_upper": np.percentile(boot_ds, 97.5)
    }


def leave_one_out(indices_df, tier1_models):
    """Leave-one-out sensitivity: does removing any model flip significance?"""
    results = []
    all_models = indices_df.model.values
    for remove_model in all_models:
        subset = indices_df[indices_df.model != remove_model]
        valid = subset[~subset.model.isin(tier1_models)]
        invalid = subset[subset.model.isin(tier1_models)]
        if len(invalid) < 2:
            continue
        t, p = stats.ttest_ind(valid.item_sensitivity, invalid.item_sensitivity)
        results.append({
            "removed": remove_model,
            "t": t, "p": p,
            "significant": p < 0.05
        })
    return pd.DataFrame(results)


def threshold_sweep(indices_df, L_range=None, F_range=None):
    """Sweep Tier 1 thresholds and report classification stability."""
    if L_range is None:
        L_range = [0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    if F_range is None:
        F_range = [0.30, 0.40, 0.50, 0.60]
    
    results = []
    for L_thresh in L_range:
        for F_thresh in F_range:
            flagged = []
            for _, r in indices_df.iterrows():
                if (r.RBS > 0 or r.L >= L_thresh or
                    r.F >= F_thresh or r.Fp >= F_thresh):
                    flagged.append(r.model)
            results.append({
                "L_threshold": L_thresh,
                "F_threshold": F_thresh,
                "n_flagged": len(flagged),
                "models": flagged
            })
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Run robustness.py after compute_indices.py for full results.")
    print("See manuscript Section 3.5 for details.")
