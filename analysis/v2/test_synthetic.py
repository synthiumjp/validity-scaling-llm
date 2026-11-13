"""Synthetic policies run through the actual v0.2 tool (not hand-computed).

524 items, 466 correct / 58 incorrect (accuracy .889, the battery median).
Policies are constructed deterministically so results do not depend on a seed.
"""
import numpy as np
from validity_screen import screen

N, NC = 524, 466
correct = np.r_[np.ones(NC, int), np.zeros(N - NC, int)]


def keep_frac(frac_correct, frac_incorrect):
    """KEEP on the first frac of correct items and first frac of incorrect items."""
    kc = int(round(frac_correct * NC))
    ki = int(round(frac_incorrect * (N - NC)))
    return np.r_[np.ones(kc), np.zeros(NC - kc), np.ones(ki), np.zeros(N - NC - ki)].astype(int)


POLICIES = {
    "Always KEEP":              (keep_frac(1.0, 1.0), "Invalid"),
    "Always WITHDRAW":          (keep_frac(0.0, 0.0), "Invalid"),
    "Random 50% KEEP":          (keep_frac(0.5, 0.5), "Invalid"),
    "Random 80% KEEP":          (keep_frac(0.8, 0.8), "Indeterminate"),  # v0.3: no detectable sensitivity
    "Perfect monitor":          (keep_frac(1.0, 0.0), "Valid"),
    "Noisy monitor (80/40)":    (keep_frac(0.8, 0.4), "Valid"),
    "Inverted monitor":         (keep_frac(0.0, 1.0), "Invalid"),
    "Weakly inverted (85/95)":  (keep_frac(0.85, 0.95), "Invalid"),
}


def run(verbose=True):
    rows, ok = [], True
    for name, (conf, expected) in POLICIES.items():
        r = screen(correct, conf, model_name=name)
        match = r.tier == expected
        ok &= match
        rows.append((name, r.L.value, r.Fp.value, r.rbs.value, r.auroc_binary, r.tier, expected, match))
        if verbose:
            print(f"{name:26s} L={r.L.value:.3f} Fp={r.Fp.value:.3f} RBS={r.rbs.value:+.3f} "
                  f"AUROCbin={r.auroc_binary:.3f} tier={r.tier:13s} expected={expected:13s} "
                  f"{'OK' if match else 'MISMATCH'}")
    return rows, ok


def test_synthetic():
    _, ok = run(verbose=False)
    assert ok


def test_identities():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = 400
        cor = (rng.random(n) < rng.uniform(.6, .95)).astype(int)
        conf = np.where(rng.random(n) < rng.uniform(.3, .9), cor, 1 - cor)
        conf = np.where(rng.random(n) < .5, 1, conf)
        r = screen(cor, conf)
        if np.isnan(r.rbs.value):
            continue
        from sklearn.metrics import roc_auc_score
        assert abs(r.auroc_binary - roc_auc_score(cor, conf)) < 1e-12
        assert abs(r.youden_j + r.rbs.value) < 1e-12


if __name__ == "__main__":
    _, ok = run()
    test_identities()
    print("all synthetic policies as expected:", ok, "| identities hold")


# ---------------------------------------------------------------- stochastic policies (paper Table 2)
STOCH = [  # name, P(KEEP | correct), P(KEEP | incorrect)
    ("Always KEEP", 1.00, 1.00), ("Always WITHDRAW", 0.00, 0.00),
    ("Random, 50% KEEP", 0.50, 0.50), ("Random, 80% KEEP", 0.80, 0.80), ("Random, 95% KEEP", 0.95, 0.95),
    ("Near-blanket confidence (97/97)", 0.97, 0.97),
    ("Perfect monitor", 1.00, 0.00), ("Noisy monitor (80/40)", 0.80, 0.40), ("Weak monitor (90/75)", 0.90, 0.75),
    ("Weakly inverted (85/95)", 0.85, 0.95), ("Inverted (20/80)", 0.20, 0.80),
]


def simulate(n_correct=460, n_incorrect=40, draws=2000, seed=7):
    """Tier distribution over random draws of each policy (KEEP drawn independently per item)."""
    rng = np.random.default_rng(seed)
    cor = np.r_[np.ones(n_correct, int), np.zeros(n_incorrect, int)]
    out = []
    for name, pc, pi in STOCH:
        tiers = {"Valid": 0, "Indeterminate": 0, "Invalid": 0, "Uncertifiable": 0}
        for _ in range(draws):
            keep = np.r_[rng.random(n_correct) < pc, rng.random(n_incorrect) < pi].astype(int)
            tiers[screen(cor, keep).tier] += 1
        out.append((name, pc, pi, {k: v / draws for k, v in tiers.items()}))
    return out
