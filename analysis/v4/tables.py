"""LaTeX table bodies generated from the outputs (no hand-typed values).

Run: python3 -I tables.py <out_dir> <paper_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import validity_screen as vs

OUT, PAPER = Path(sys.argv[1]), Path(sys.argv[2])


def f3(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    s = s.replace("0.", ".", 1) if not signed else s.replace("+0.", "+.").replace("-0.", "$-$.")
    return s


def ci(lo, hi, signed=False):
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return ""
    return f" [{f3(lo, signed)}, {f3(hi, signed)}]"


REASON = {"signal_detected": "", "no_detectable_sensitivity": "no detectable sensitivity",
          "blanket_confidence_suspected": "blanket confidence suspected", "too_few_errors": "fewer than 10 errors",
          "blanket_confidence": "blanket confidence", "blanket_withdrawal": "blanket withdrawal",
          "inversion": "inversion", "inversion_suspected": "inversion suspected",
          "blanket_withdrawal_suspected": "blanket withdrawal suspected", "too_few_correct": "fewer than 10 correct"}
FLAG = {"low_coverage": "LC", "response_set_not_excluded": "RS", "fixed_responding": "FR"}

# ---------------- battery (appendix)
C = pd.read_csv(OUT / "cells.csv")
B = C[C.ds == "battery"].copy()
order = {"Invalid": 0, "Uncertifiable": 1, "Indeterminate": 2, "Valid": 3}
B = B.assign(o=B.tier.map(order)).sort_values(["o", "J"], ascending=[True, True])
lines = []
for _, r in B.iterrows():
    flags = ",".join(FLAG[f] for f in str(r["flags"]).split("|") if f in FLAG)
    tier = r.tier + (f" ({REASON[r.reason]})" if REASON.get(r.reason) else "")
    lines.append(" & ".join([
        r.unit, str(int(r.n_err)), f3(r.accuracy),
        f3(r.L) + ci(r.L_lo, r.L_hi), f3(r.Fp) + ci(r.Fp_lo, r.Fp_hi),
        f3(r.J, True) + ci(r.J_lo, r.J_hi, True),
        "--" if np.isnan(r.p_signal) else (f"{r.p_signal:.3f}".replace("0.", ".", 1) if r.p_signal >= .001 else "$<$.001"),
        flags or "", tier]) + r" \\")
(PAPER / "tab_battery.tex").write_text("\n".join(lines) + "\n")

# ---------------- planning table: errors to certify, power, items for 90% power
rows = []
for L in [1.0, .99, .98, .97, .96, .95]:
    pw = [vs.power_invalid(L, e) for e in [35, 71, 150, 500]]
    need = [vs.items_needed(a, L, .9) for a in [.80, .90, .95]]
    rows.append(" & ".join([f"{L:.2f}".lstrip("0")] + [f"{p:.2f}".replace("0.", ".", 1) if p < 1 else "1" for p in pw]
                           + [f"{n:,}" if n else "never" for n in need]) + r" \\")
(PAPER / "tab_planning.tex").write_text("\n".join(rows) + "\n")
(PAPER / "tab_errors_needed.tex").write_text(
    " & ".join(str(vs.errors_needed(k)) for k in range(8)) + r" \\" + "\n")

# ---------------- held-out (if available)
H = OUT / "heldout_cells.csv"
if H.exists():
    Hd = pd.read_csv(H)
    lines = []
    for _, r in Hd.sort_values(["condition", "model"], ascending=[False, True]).iterrows():
        tier = r.tier + (f" ({REASON[r.reason]})" if REASON.get(r.reason) else "")
        fl = ",".join(FLAG[f] for f in str(r["flags"]).split("|") if f in FLAG)
        tier += f"; {fl}" if fl else ""
        lines.append(" & ".join([r.condition, r.model, f"{r.parse_rate:.2f}".replace("0.", ".", 1),
                                 str(int(r.n_scored)), str(int(r.n_err)), f3(r.accuracy),
                                 "--" if np.isnan(r.share_ge95) else f"{r.share_ge95:.2f}".replace("0.", ".", 1),
                                 (f3(r.L) + f" ({f3(r.L_lo)})") if not np.isnan(r.L) else "--", f3(r.J, True) + ci(r.J_lo, r.J_hi, True), f3(r.auroc_raw),
                                 tier + ("" if r.confirmatory else r"$^\dagger$")]) + r" \\")
    (PAPER / "tab_heldout.tex").write_text("\n".join(lines) + "\n")
print("tables written")
