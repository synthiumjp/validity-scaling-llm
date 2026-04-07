"""
Synthetic policy validation for the validity tier system.
Generates 8 synthetic response profiles and confirms correct classification.

Author: Jon-Paul Cacioli
Date: April 2026
"""

import numpy as np
from compute_indices import compute_validity_indices, classify_tier1


def generate_synthetic_policies(item_accuracies, n_items, n_iterations=1000):
    """Generate 8 synthetic response policies on real item structure."""
    
    policies = {}
    
    # 1. Always KEEP + BET
    policies["Always KEEP+BET"] = {
        "keep": np.ones(n_items, dtype=int),
        "bet": np.ones(n_items, dtype=int)
    }
    
    # 2. Always WITHDRAW + NO BET
    policies["Always WITHDRAW+NOBET"] = {
        "keep": np.zeros(n_items, dtype=int),
        "bet": np.zeros(n_items, dtype=int)
    }
    
    # 3. Random 50/50
    policies["Random 50/50"] = {
        "keep": (np.random.rand(n_items) > 0.5).astype(int),
        "bet": (np.random.rand(n_items) > 0.5).astype(int)
    }
    
    # 4. Random 80% KEEP
    policies["Random 80% KEEP"] = {
        "keep": (np.random.rand(n_items) > 0.2).astype(int),
        "bet": (np.random.rand(n_items) > 0.3).astype(int)
    }
    
    # 5. Perfect monitor
    correct = (np.random.rand(n_items) < item_accuracies).astype(int)
    policies["Perfect monitor"] = {
        "keep": correct,
        "bet": correct
    }
    
    # 6. Noisy monitor (80/60)
    noisy_keep = np.where(correct,
        (np.random.rand(n_items) > 0.2).astype(int),
        (np.random.rand(n_items) > 0.6).astype(int))
    policies["Noisy monitor (80/60)"] = {
        "keep": noisy_keep,
        "bet": noisy_keep.copy()
    }
    
    # 7. Inverted monitor
    policies["Inverted monitor"] = {
        "keep": (1 - correct),
        "bet": (1 - correct)
    }
    
    # 8. R1-like (inverted KEEP + always BET)
    policies["R1-like"] = {
        "keep": (1 - correct),
        "bet": np.ones(n_items, dtype=int)
    }
    
    return policies, correct


if __name__ == "__main__":
    print("Run synthetic_validation.py for full simulation results.")
    print("See manuscript Section 2.5 and 3.3 for details.")
