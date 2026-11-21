"""Figures for the v0.4 paper.

Run: python3 -I figures.py <out_dir> <fig_dir>
Palette: categorical slots 1-5 of the reference palette, validated in fixed order
(adjacent CVD dE >= 9.1, normal-vision dE >= 19.6); contrast relief by legends,
direct labels and the tables in the paper.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402
import pandas as pd  # noqa: E402

import validity_screen as vs  # noqa: E402

OUT, FIG = Path(sys.argv[1]), Path(sys.argv[2])
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # blue ramp, ordinal from step 250
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.6,
})

# Tier categories (fixed order: Valid, Indet. no signal, Indet. suspected, Invalid, Uncertifiable)
CATS = ["Valid", "Indeterminate: no detectable sensitivity", "Indeterminate: response set or inversion suspected",
        "Invalid", "Uncertifiable"]
SHORT = ["Valid", "Indet.: no signal", "Indet.: suspected", "Invalid", "Uncertifiable"]


def cat_of(row):
    if row.tier == "Indeterminate":
        return CATS[1] if row.reason == "no_detectable_sensitivity" else CATS[2]
    return {"Valid": CATS[0], "Invalid": CATS[3], "Uncertifiable": CATS[4]}[row.tier]


C = pd.read_csv(OUT / "cells.csv")
C["tcat"] = C.apply(cat_of, axis=1)

# ---------------------------------------------------------------- Fig 2: operating characteristics
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.25), constrained_layout=True)

# (a) false-Valid rate at 452 items, by error count: Newcombe vs mid-p (max over keep rates .05 to .99)
CP = pd.read_csv(OUT / "oc_size_compare.csv")
g = CP.groupby("errors")[["newcombe", "midp"]].max().sort_index()
ax[0].plot(g.index, g.newcombe, color=SLOT[1], marker="o", ms=3.5)
ax[0].plot(g.index, g.midp, color=SLOT[0], marker="s", ms=3.5)
ax[0].axhline(.025, color=INK2, lw=0.8, ls="--")
ax[0].text(52, .027, "nominal .025", color=INK2, fontsize=6.5, ha="right", va="bottom")
ax[0].set(xlabel="Errors (452 items)", ylabel="False-Valid rate (worst case)", ylim=(0, .1))
ax[0].text(12, .089, "Newcombe (v0.3)", color=INK, fontsize=6.5, ha="left", va="bottom")
ax[0].text(52, .016, "mid-p (v0.4)", color=INK, fontsize=6.5, ha="right", va="top")
ax[0].set_title("a  No signal: false Valid", loc="left", fontsize=8, color=INK)

# (b) P(Valid) by items for J = .05, .10, .20 at accuracy .90 (KEEP .95 on correct)
P = pd.read_csv(OUT / "oc_power_valid.csv")
for i, J in enumerate([.05, .10, .20]):
    s = P[(P.accuracy == .90) & (P.keep_correct == .95) & (P.J == J)].sort_values("N")
    ax[1].plot(s.N, s.p_valid, color=SEQ[2 * i], marker="o", ms=3)
    yv = float(s[s.N == 452].p_valid.iloc[0])
    ax[1].text(430, yv + 0.07, f"J = {J:.2f}", fontsize=6.5, color=INK, va="center", ha="right")
ax[1].set_xscale("log"); ax[1].xaxis.set_minor_locator(NullLocator()); ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[1].set_xticks([226, 452, 1000, 3000]); ax[1].set_xticklabels(["226", "452", "1k", "3k"])
ax[1].set(xlabel="Items (accuracy .90)", ylabel="P(Valid)", ylim=(0, 1.03), xlim=(200, 3600))
ax[1].set_title("b  Real signal: power", loc="left", fontsize=8, color=INK)

# (c) P(certify blanket confidence) by errors, true L
PI = pd.read_csv(OUT / "oc_power_invalid.csv")
Ls = [1.0, .99, .98, .97, .96, .95]
for i, L in enumerate(Ls):
    s = PI[PI.true_L == L].sort_values("errors")
    ax[2].plot(s.errors, s.p_certify, color=SEQ[len(Ls) - 1 - i], marker="o", ms=2.5)
    if L in (.97, .96, .95):
        ax[2].text(s.errors.iloc[-1] * 1.08, s.p_certify.iloc[-1] + {.97: 0.0, .96: -0.08, .95: 0.0}[L], f"L = {L:.2f}", fontsize=6, color=INK, va="center")
    if L == .99:
        ax[2].text(1080, 1.04, "L = .98-1.00", fontsize=6, color=INK, va="bottom")
ax[2].set_xscale("log"); ax[2].xaxis.set_minor_locator(NullLocator()); ax[2].xaxis.set_minor_formatter(NullFormatter())
ax[2].set_xticks([35, 100, 300, 1000]); ax[2].set_xticklabels(["35", "100", "300", "1k"])
ax[2].set(xlabel="Errors", ylabel="P(certify blanket confidence)", ylim=(0, 1.1), xlim=(30, 3000))
ax[2].set_title("c  Certifying a response set", loc="left", fontsize=8, color=INK)

# (d) two-administration agreement by items (battery models)
AG = pd.read_csv(OUT / "oc_agreement.csv")
q = AG.groupby("N").agree.describe()
ax[3].fill_between(q.index, q["25%"], q["75%"], color=SEQ[0], alpha=.45, lw=0, label="middle half of models")
ax[3].plot(q.index, q["mean"], color=SEQ[3], marker="o", ms=3, label="mean")
ax[3].set_xscale("log"); ax[3].xaxis.set_minor_locator(NullLocator()); ax[3].xaxis.set_minor_formatter(NullFormatter())
ax[3].set_xticks([226, 452, 1000, 2000, 5000])
ax[3].set_xticklabels(["226", "452", "1k", "2k", "5k"])
ax[3].set(xlabel="Items", ylabel="P(two runs agree)", ylim=(.3, 1.02))
ax[3].legend(loc="lower right", fontsize=6.5)
ax[3].set_title("d  Verdict stability", loc="left", fontsize=8, color=INK)
fig.savefig(FIG / "fig2_operating.pdf"); fig.savefig(FIG / "fig2_operating.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: verdicts by cut point
panels = [("atlas", "Atlas MMLU: 33 models", None), ("mmlu500", "MMLU-500: 20 models", None),
          ("yang", "Yang et al.: 110 cells", "basic_1s")]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, constrained_layout=True)
cuts = [50, 70, 80, 90, 95, 100]
for k, (ds, title, method) in enumerate(panels):
    S = C[C.ds == ds]
    if method:
        S = S[S.method == method]
    tab = pd.crosstab(S.cut, S.tcat, normalize="index").reindex(columns=CATS, fill_value=0).reindex(cuts)
    bottom = np.zeros(len(cuts))
    for j, cat in enumerate(CATS):
        v = tab[cat].values
        if v.sum() == 0:
            continue
        ax[k].bar(range(len(cuts)), v, bottom=bottom, color=SLOT[j], width=.72, edgecolor=SURF, linewidth=1.0,
                  label=SHORT[j])
        bottom += v
    ax[k].set_xticks(range(len(cuts))); ax[k].set_xticklabels(cuts)
    ax[k].set(xlabel="Cut point (confidence ≥)", ylim=(0, 1)); ax[k].grid(axis="x", visible=False)
    ax[k].set_title(title, loc="left", fontsize=8, color=INK)
ax[0].set_ylabel("Share of units")
h, l = ax[2].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=7)
fig.savefig(FIG / "fig3_cutpoints.pdf"); fig.savefig(FIG / "fig3_cutpoints.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: battery J with intervals
B = C[C.ds == "battery"].copy().sort_values("J", na_position="first")
fig, ax = plt.subplots(figsize=(3.5, 3.6), constrained_layout=True)
mk = {CATS[0]: "o", CATS[1]: "s", CATS[2]: "D", CATS[3]: "X", CATS[4]: "^"}
y = np.arange(len(B))
for i, (_, r) in enumerate(B.iterrows()):
    j = SLOT[CATS.index(r.tcat)]
    if np.isnan(r.J):
        ax.plot(-0.085, i, marker=mk[r.tcat], color=SLOT[CATS.index(r.tcat)], ms=5, mec=SURF, mew=0.8)
        ax.text(-0.06, i, "J not estimable (6 errors)", fontsize=6.5, color=INK2, va="center")
        continue
    ax.plot([r.J_lo, r.J_hi], [i, i], color=j, lw=1.4)
    ax.plot(r.J, i, marker=mk[r.tcat], color=j, ms=5, mec=SURF, mew=0.8)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(B.unit, fontsize=6.5)
ax.set(xlabel="Item sensitivity J (95% Newcombe interval)", xlim=(-.1, .6), ylim=(-0.7, len(B) - 0.3))
ax.grid(axis="y", visible=False)
for cat in [CATS[0], CATS[1], CATS[2], CATS[4]]:
    ax.plot([], [], marker=mk[cat], ls="", color=SLOT[CATS.index(cat)], label=SHORT[CATS.index(cat)])
ax.legend(loc="lower right", fontsize=6.5)
fig.savefig(FIG / "fig4_battery.pdf"); fig.savefig(FIG / "fig4_battery.png", dpi=200)
plt.close(fig)
print("figures written to", FIG)
