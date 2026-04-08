#!/usr/bin/env python3
"""
Test suite for validity_screen.py.

Validates the screening protocol against known paper results and
synthetic response policies. All expected values are from:

    Cacioli (2026d). Before You Interpret the Profile.
    Cacioli (2026e). Screen Before You Interpret.

Run from the screen/ directory:
    python test_validity_screen.py

Or from repo root:
    python screen/test_validity_screen.py

Exit code 0 = all tests passed. Non-zero = assertion failure.
"""

import numpy as np
import sys
from pathlib import Path

# Resolve imports relative to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validity_screen import screen, summary_table, binarise


# ============================================================
# Helper: reconstruct item-level vectors from summary stats
# ============================================================

def make_data(n_correct, n_incorrect, L, Fp):
    """
    Build correct/confidence arrays from summary statistics.

    The 2x2 table cells are:
        a = correct + high conf = n_correct - c
        b = incorrect + high conf = round(L * n_incorrect)
        c = correct + low conf = round(Fp * n_correct)
        d = incorrect + low conf = n_incorrect - b

    Minor rounding artefacts (±1 item) are expected when
    reconstructing from rounded proportions.
    """
    b = round(L * n_incorrect)
    d = n_incorrect - b
    c = round(Fp * n_correct)
    a = n_correct - c

    correct = np.array([True] * n_correct + [False] * n_incorrect)
    confidence = np.array([True] * a + [False] * c + [True] * b + [False] * d)
    return correct, confidence


# ============================================================
# Test 1: Known models from the paper
# ============================================================

def test_known_models():
    """
    Verify screen output against known paper values.

    Expected tiers are from the derivation study (Cacioli, 2026d).
    L and Fp values should reconstruct within ±0.005 of targets
    (rounding tolerance from integer cell counts).

    Models with very few errors (Qwen 80B Think, Gemini 3.1 Pro)
    produce min_cell < 5 and return "Insufficient data". This is
    correct behaviour. The screen's cell-count guard is working.
    """
    print("=" * 60)
    print("TEST 1: Known models from the paper")
    print("=" * 60)

    models = {
        # (n_correct, n_incorrect, L, Fp, expected_tier)
        "Claude Haiku 4.5": (467, 57, 0.509, 0.176, "Valid"),
        "Qwen 80B Think":   (485, 39, 0.974, 0.008, "Insufficient data"),
        "DeepSeek-R1":      (447, 77, 0.922, 0.946, "Invalid"),
        "Gemini 3.1 Pro":   (506, 18, 0.967, 0.034, "Insufficient data"),
        "Gemma 3 1B":       (365, 159, 0.731, 0.297, "Indeterminate"),
        "Gemma 12B":        (449, 75, 0.493, 0.258, "Valid"),
        "GPT-5.4 nano":     (452, 72, 0.486, 0.369, "Valid"),
        "Sonnet 4.6":       (494, 30, 0.763, 0.029, "Valid"),
    }

    results = []
    n_passed = 0
    for name, (nc, ni, target_L, target_Fp, expected_tier) in models.items():
        correct, confidence = make_data(nc, ni, target_L, target_Fp)
        r = screen(correct, confidence, model_name=name, benchmark_name="Classical Minds v1")
        results.append(r)

        # Check tier classification
        assert r.tier == expected_tier, (
            f"FAIL: {name} tier = {r.tier}, expected {expected_tier}"
        )

        # Check index values (only if screen computed them)
        if r.L is not None:
            assert abs(r.L.value - target_L) < 0.005, (
                f"FAIL: {name} L = {r.L.value:.3f}, expected {target_L:.3f}"
            )
        if r.Fp is not None:
            assert abs(r.Fp.value - target_Fp) < 0.005, (
                f"FAIL: {name} Fp = {r.Fp.value:.3f}, expected {target_Fp:.3f}"
            )

        status = "OK"
        L_str = f"L={r.L.value:.3f}" if r.L else "L=N/A"
        Fp_str = f"Fp={r.Fp.value:.3f}" if r.Fp else "Fp=N/A"
        print(f"  {status}  {name:<25} {L_str}  {Fp_str}  tier={r.tier}")
        n_passed += 1

    print(f"\n  {n_passed}/{len(models)} models passed.\n")
    print(summary_table(results))
    return results


# ============================================================
# Test 2: VRS Table formatting
# ============================================================

def test_vrs_table(results):
    """
    Verify VRS Table output contains all required fields.

    The VRS Table is the mandated reporting format from
    Cacioli (2026e) Section 2.8.
    """
    print("\n" + "=" * 60)
    print("TEST 2: VRS Table formatting")
    print("=" * 60)

    r1 = [r for r in results if "R1" in r.model_name][0]
    table = r1.vrs_table()

    required_fields = [
        "Model", "Benchmark", "N items", "Accuracy",
        "2x2 table", "TRIN", "L", "Fp", "RBS",
        "r(confidence, correct)", "Tier classification", "Flagging reason"
    ]
    for field in required_fields:
        assert field in table, f"FAIL: VRS Table missing '{field}'"

    assert "Invalid" in table, "FAIL: R1 should be classified Invalid"
    assert "0.946" in table, "FAIL: R1 Fp should be 0.946"
    assert "-0.798" in table, "FAIL: R1 r should be -0.798"

    print("  OK  VRS Table contains all required fields.")
    print("  OK  R1 values correct in VRS Table.\n")
    print(table)
    print()


# ============================================================
# Test 3: Synthetic response policies
# ============================================================

def test_synthetic_policies():
    """
    Validate the screen against synthetic response policies.

    Ground-truth validity is known by construction:
    - Uninformative policies should NOT return "Valid"
    - The noisy monitor has genuine discrimination and should be "Valid"
    - Degenerate policies (all KEEP, all WITHDRAW, perfect, inverted)
      produce zero cells and should return "Insufficient data"
    """
    print("=" * 60)
    print("TEST 3: Synthetic response policies")
    print("=" * 60)

    np.random.seed(42)
    n = 524
    n_correct = round(n * 0.90)
    correct = np.array([True] * n_correct + [False] * (n - n_correct))

    # Always KEEP: d=0 (no incorrect items with low confidence)
    r = screen(correct, np.ones(n, dtype=bool), model_name="Always KEEP")
    assert r.tier == "Insufficient data", f"FAIL: Always KEEP tier = {r.tier}"
    print(f"  OK  Always KEEP          -> {r.tier}")

    # Always WITHDRAW: a=0 (no correct items with high confidence)
    r = screen(correct, np.zeros(n, dtype=bool), model_name="Always WITHDRAW")
    assert r.tier == "Insufficient data", f"FAIL: Always WITHDRAW tier = {r.tier}"
    print(f"  OK  Always WITHDRAW      -> {r.tier}")

    # Random 50/50: Fp near 0.50, should not pass as Valid
    conf_rand = np.random.binomial(1, 0.5, n).astype(bool)
    r = screen(correct, conf_rand, model_name="Random 50/50")
    assert r.tier != "Valid", f"FAIL: Random 50/50 should not be Valid, got {r.tier}"
    print(f"  OK  Random 50/50         -> {r.tier}")

    # Perfect monitor: b=0, d=0 (never confident on errors, never unconfident on correct)
    r = screen(correct, correct.copy(), model_name="Perfect monitor")
    assert r.tier == "Insufficient data", f"FAIL: Perfect monitor tier = {r.tier}"
    print(f"  OK  Perfect monitor      -> {r.tier}")

    # Noisy monitor: genuine discrimination, should be Valid
    conf_noisy = np.zeros(n, dtype=bool)
    for i in range(n):
        if correct[i]:
            conf_noisy[i] = np.random.random() < 0.80
        else:
            conf_noisy[i] = np.random.random() < 0.40
    r = screen(correct, conf_noisy, model_name="Noisy monitor")
    assert r.tier == "Valid", f"FAIL: Noisy monitor should be Valid, got {r.tier}"
    assert r.r_conf_correct.value > 0, "FAIL: Noisy monitor r should be positive"
    print(f"  OK  Noisy monitor        -> {r.tier} (r = {r.r_conf_correct.value:+.3f})")

    # Inverted monitor: a=0, d=0
    r = screen(correct, ~correct, model_name="Inverted monitor")
    assert r.tier == "Insufficient data", f"FAIL: Inverted monitor tier = {r.tier}"
    print(f"  OK  Inverted monitor     -> {r.tier}")

    print(f"\n  All 6 synthetic policies behaved as expected.\n")


# ============================================================
# Test 4: Binarise utility
# ============================================================

def test_binarise():
    """
    Verify the binarise() convenience function.

    Fixed threshold: values >= threshold map to True.
    Median method: uses sample median as threshold.
    """
    print("=" * 60)
    print("TEST 4: Binarise utility")
    print("=" * 60)

    cont = np.array([0.2, 0.4, 0.6, 0.8, 0.9])

    # Fixed threshold at 0.5
    result_fixed = binarise(cont, 0.5)
    expected_fixed = np.array([False, False, True, True, True])
    assert np.array_equal(result_fixed, expected_fixed), (
        f"FAIL: Fixed binarise got {result_fixed}"
    )
    print(f"  OK  Fixed threshold: {cont} -> {result_fixed}")

    # Median method (median = 0.6)
    result_median = binarise(cont, method='median')
    expected_median = np.array([False, False, True, True, True])
    assert np.array_equal(result_median, expected_median), (
        f"FAIL: Median binarise got {result_median}"
    )
    print(f"  OK  Median method:  {cont} -> {result_median}")
    print()


# ============================================================
# Test 5: Edge cases
# ============================================================

def test_edge_cases():
    """
    Verify the screen handles degenerate inputs gracefully.
    """
    print("=" * 60)
    print("TEST 5: Edge cases")
    print("=" * 60)

    # Mismatched lengths should raise ValueError
    try:
        screen(np.array([True, False]), np.array([True]))
        assert False, "FAIL: Should have raised ValueError"
    except ValueError:
        print(f"  OK  Mismatched lengths   -> raised ValueError")

    # All correct (n_incorrect = 0): L undefined, should handle gracefully
    all_correct = np.ones(100, dtype=bool)
    conf = np.random.binomial(1, 0.7, 100).astype(bool)
    r = screen(all_correct, conf, model_name="All correct")
    assert r.tier == "Insufficient data", f"FAIL: All correct tier = {r.tier}"
    print(f"  OK  All correct (n_inc=0) -> {r.tier}")

    # All incorrect (n_correct = 0): Fp undefined
    all_incorrect = np.zeros(100, dtype=bool)
    r = screen(all_incorrect, conf, model_name="All incorrect")
    assert r.tier == "Insufficient data", f"FAIL: All incorrect tier = {r.tier}"
    print(f"  OK  All incorrect (n_cor=0) -> {r.tier}")

    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print()
    results = test_known_models()
    test_vrs_table(results)
    test_synthetic_policies()
    test_binarise()
    test_edge_cases()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
