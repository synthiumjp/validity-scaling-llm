"""Agreement between validity-screen and an independent reimplementation written from the
paper's text by a separate reviewer (independent/myscreen.py), on every table with cells
drawn from a grid that includes zero cells and the MIN_N and certification boundaries.
Run: python3 -I check_agreement.py"""
import itertools
import sys
from pathlib import Path

import numpy as np

import validity_screen as vs

sys.path.append(str(Path(__file__).resolve().parent / "independent"))
import myscreen as ms  # noqa: E402

vals = [0, 1, 2, 3, 5, 9, 10, 11, 19, 20, 34, 35, 36, 50, 95, 100, 133, 140, 300]
n = bad = 0
for a, b, c, d in itertools.product(vals, repeat=4):
    if a + b + c + d == 0:
        continue
    cor = np.r_[np.ones(a + c), np.zeros(b + d)]
    kp = np.r_[np.ones(a), np.zeros(c), np.ones(b), np.zeros(d)]
    r = vs.screen(cor, kp)
    n += 1
    if (r.tier, r.reason_code) != ms.tier(a, b, c, d):
        bad += 1
        if bad <= 5:
            print("MISMATCH", (a, b, c, d), r.tier, r.reason_code, ms.tier(a, b, c, d))
print(f"tables {n}; tier or reason mismatches {bad}")
