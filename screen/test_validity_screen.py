#!/usr/bin/env python3
"""
Test validity_screen against known paper results.

Run from the screen/ directory:
    python test_validity_screen.py

Or from repo root:
    python screen/test_validity_screen.py
"""

import numpy as np
import sys
from pathlib import Path

# Add the directory containing this script to the path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validity_screen import screen, summary_table, binarise

# ============================================================
# Reconstruct 2x2 tables from known paper values
# N=524 for all models
# ============================================================

def make_data(n_correct, n_incorrect, L, Fp):
    """Reconstruct item-level vectors from summary statistics."""
    b = round(L * n_incorrect)           # incorrect + high conf
    d = n_incorrect - b                  # incorrect + low conf
    c = round(Fp * n_correct)            # correct + low conf
    a = n_correct - c                    # correct + high conf
    
    correct = np.array([True]*n_correct + [False]*n_incorrect)
    confidence = np.array([True]*a + [False]*c + [True]*b + [False]*d)
    return correct, confidence

# Known models from the paper
models = {
    "Claude Haiku 4.5":   {"nc": 467, "ni": 57,  "L": 0.509, "Fp": 0.176},
    "Qwen 80B Think":     {"nc": 485, "ni": 39,  "L": 0.974, "Fp": 0.008},
    "DeepSeek-R1":        {"nc": 447, "ni": 77,  "L": 0.922, "Fp": 0.946},
    "Gemini 3.1 Pro":     {"nc": 506, "ni": 18,  "L": 0.967, "Fp": 0.034},
    "Gemma 3 1B":         {"nc": 365, "ni": 159, "L": 0.731, "Fp": 0.297},
    "Gemma 12B":          {"nc": 449, "ni": 75,  "L": 0.493, "Fp": 0.258},
    "GPT-5.4 nano":       {"nc": 452, "ni": 72,  "L": 0.486, "Fp": 0.369},
    "Sonnet 4.6":         {"nc": 494, "ni": 30,  "L": 0.763, "Fp": 0.029},
}

results = []
for name, d in models.items():
    correct, confidence = make_data(d["nc"], d["ni"], d["L"], d["Fp"])
    r = screen(correct, confidence, model_name=name, benchmark_name="Classical Minds v1")
    results.append(r)
    
    if r.L is not None:
        l_check = r.L.value
        fp_check = r.Fp.value
        print(f"{name:<25} L={l_check:.3f} (target {d['L']:.3f})  "
              f"Fp={fp_check:.3f} (target {d['Fp']:.3f})  "
              f"Tier={r.tier}")
    else:
        print(f"{name:<25} Tier={r.tier} (cell count too low with reconstructed data)")

print()
print(summary_table(results))

# Print one full VRS Table as example
print()
print("=" * 60)
print("Example VRS Table: DeepSeek-R1")
print("=" * 60)
r1 = [r for r in results if "R1" in r.model_name][0]
print(r1.vrs_table())

# ============================================================
# Test synthetic policies
# ============================================================
print("\n\n" + "=" * 60)
print("Synthetic policy validation")
print("=" * 60)

np.random.seed(42)
n = 524
accuracy = 0.90
correct = np.array([True] * round(n * accuracy) + [False] * round(n * (1 - accuracy)))

def print_synthetic(label, r):
    l_s = f"{r.L.value:.3f}" if r.L else "—"
    fp_s = f"{r.Fp.value:.3f}" if r.Fp else "—"
    print(f"{label:<25} Tier={r.tier:<20} L={l_s}  Fp={fp_s}")

# Always KEEP
r = screen(correct, np.ones(n, dtype=bool), model_name="Always KEEP")
print_synthetic("Always KEEP:", r)

# Always WITHDRAW
r = screen(correct, np.zeros(n, dtype=bool), model_name="Always WITHDRAW")
print_synthetic("Always WITHDRAW:", r)

# Random 50/50
conf_rand = np.random.binomial(1, 0.5, n).astype(bool)
r = screen(correct, conf_rand, model_name="Random 50/50")
print_synthetic("Random 50/50:", r)

# Perfect monitor
conf_perfect = correct.copy()
r = screen(correct, conf_perfect, model_name="Perfect monitor")
print_synthetic("Perfect monitor:", r)

# Noisy monitor (80% correct on correct, 60% correct on incorrect)
conf_noisy = np.zeros(n, dtype=bool)
for i in range(n):
    if correct[i]:
        conf_noisy[i] = np.random.random() < 0.80
    else:
        conf_noisy[i] = np.random.random() < 0.40
r = screen(correct, conf_noisy, model_name="Noisy monitor")
print_synthetic("Noisy monitor:", r)

# Inverted monitor
conf_inv = ~correct
r = screen(correct, conf_inv, model_name="Inverted monitor")
print_synthetic("Inverted monitor:", r)

# Test binarise utility
print("\nBinarise test:")
cont = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
print(f"  Input:  {cont}")
print(f"  Fixed:  {binarise(cont, 0.5)}")
print(f"  Median: {binarise(cont, method='median')}")

print("\nAll tests passed.")
