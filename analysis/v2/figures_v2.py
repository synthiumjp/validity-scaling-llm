"""Figures for 17714 v2. Run: python figures_v2.py <outdir> <figdir> (see run_all.sh)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validity_screen import wilson_ci  # noqa: E402

OUT, FIG = Path(sys.argv[1]), Path(sys.argv[2])
FIG.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
TIER = {"Valid": ("#2a78d6", "o"), "Indeterminate": ("#eda100", "D"), "Invalid": ("#e34948", "^"),
        "Uncertifiable": ("#8a8984", "s")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def tier_legend(ax, tiers, loc="lower right"):
    h = [plt.Line2D([], [], ls="", marker=TIER[t][1], mfc=TIER[t][0], mec=INK, mew=0.5, ms=6, label=t)
         for t in tiers]
    ax.legend(handles=h, loc=loc, fontsize=8, handletextpad=0.3)


P = pd.read_csv(OUT / "per_model.csv")
C = json.loads((OUT / "cross_numbers.json").read_text())

# ---------------------------------------------------------------- Fig 1 protocol flowchart
fig, ax = plt.subplots(figsize=(6.2, 6.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")


def box(x, y, w, h, text, fc="#f3f2ee", ec=INK2, bold=False, fs=8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal",
            linespacing=1.25)


def arrow(x0, y0, x1, y1, label=None, side="right"):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    if label:
        ax.text((x0 + x1) / 2 + (0.12 if side == "right" else -0.12), (y0 + y1) / 2, label, fontsize=7,
                color=INK2, ha="left" if side == "right" else "right", va="center")


steps = [
    (12.2, "Step 1. Exclude non-answers (declines, unparseable)\nbuild 2×2 table; report abstention count"),
    (10.8, "Step 2. TRIN = max(n_high, n_low)/N\n≥ .95 → structural warning only"),
    (9.4, "Step 3. Fp = c/(a+c)   [needs ≥ 10 correct]\n≥ .50: Wilson LB > .40 → Invalid, else Indeterminate"),
    (8.0, "Step 4. L = b/(b+d)   [needs ≥ 10 errors]\n≥ .95: Wilson LB > .90 → Invalid, else Indeterminate"),
    (6.6, "Step 5. RBS = Fp − (1 − L) = −J   [needs both]\n> 0: Newcombe CI excludes 0 → Invalid, else Indeterminate"),
    (5.2, "Step 6. r(confidence, correct) = φ, with 95% CI\nreported in the VRS Table"),
]
for y, t in steps:
    box(5, y, 7.4, 1.0, t)
for (y0, _), (y1, _) in zip(steps[:-1], steps[1:]):
    arrow(5, y0 - 0.5, 5, y1 + 0.5)
ax.text(0.2, 12.9, "Stage A: validity screening (before any Stage B metric)", fontsize=9, weight="bold", color=INK)
arrow(5, 4.7, 5, 3.75)
box(5, 3.2, 7.4, 1.05, "Tier: Invalid if any index is Invalid; else Indeterminate if any is;\n"
    "else Uncertifiable if RBS cannot be estimated; Valid only if the\nRBS interval lies below 0 (otherwise Indeterminate)", fc="#ffffff", fs=7.5)
for x, t in [(1.6, "Invalid"), (3.85, "Indeterminate"), (6.15, "Uncertifiable"), (8.4, "Valid")]:
    arrow(5, 2.65, x, 1.75)
    box(x, 1.35, 2.05, 0.75, t, fc="#ffffff", ec=TIER[t][0], bold=True)
ax.text(8.4, 0.55, "→ Stage B\n(AUROC, ECE, meta-d′,\nselective prediction)", ha="center", va="top", fontsize=7,
        color=INK2)
ax.text(1.6, 0.55, "do not interpret\nStage B", ha="center", va="top", fontsize=7, color=INK2)
ax.text(5.0, 0.55, "compute Stage B, flag it;\ncollect more errors or re-elicit", ha="center", va="top",
        fontsize=7, color=INK2)
fig.savefig(FIG / "fig1_protocol.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2 battery: L vs errors + r by model
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.62))
ns = np.arange(10, 160)
need = []
for n in ns:  # smallest L whose Wilson LB exceeds .90 at n errors
    ok = [k for k in range(n + 1) if wilson_ci(k, n)[0] > 0.90]
    need.append(min(ok) / n if ok else np.nan)
need = np.array(need)
a1.fill_between(ns, np.where(np.isnan(need), 1.0, need), 1.0, color="#e34948", alpha=0.08, lw=0)
a1.plot(ns, need, color="#e34948", lw=1.2)
a1.axhline(0.95, color=INK2, ls="--", lw=0.8)
a1.text(158, 0.947, "L = .95", ha="right", va="top", fontsize=7, color=INK2)
i100 = int(np.where(ns == 110)[0][0])
a1.annotate("above red line:\nInvalid certifiable\n(Wilson LB > .90)", (110, need[i100]), (78, 0.68),
            fontsize=7, color="#b8302f", ha="left", arrowprops=dict(arrowstyle="-", color="#b8302f", lw=0.5))
q = P.dropna(subset=["L"])
for _, r in q.iterrows():
    col, mk = TIER[r.tier]
    a1.errorbar(r.n_incorrect, r.L, yerr=[[r.L - r.L_lo], [r.L_hi - r.L]], fmt="none", ecolor=col, elinewidth=0.8,
                alpha=0.7, zorder=2)
    a1.plot(r.n_incorrect, r.L, marker=mk, ls="", mfc=col, mec=INK, mew=0.5, ms=5.5, zorder=3)
for name, tx, ty in [("Gemini 3.1 Pro", 74, 0.875), ("Qwen 80B Think", 74, 0.84)]:
    r = q[q.model == name].iloc[0]
    a1.annotate(name, (r.n_incorrect, r.L), (tx, ty), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
a1.set_xlim(0, 160); a1.set_ylim(0.3, 1.005)
a1.set_xlabel("Errors scored (b + d)"); a1.set_ylabel("L = P(KEEP | incorrect)")
a1.set_title("a  Certifying blanket confidence needs errors", loc="left", fontsize=9, color=INK)
grid(a1)
tier_legend(a1, ["Valid", "Indeterminate"], loc="lower right")

Q = P.sort_values("r").reset_index(drop=True)
for i, r in Q.iterrows():
    col, mk = TIER[r.tier]
    a2.plot([r.r_lo, r.r_hi], [i, i], color=col, lw=1.0, zorder=2)
    a2.plot(r.r, i, marker=mk, mfc=col, mec=INK, mew=0.5, ms=5.5, ls="", zorder=3)
a2.axvline(0, color=INK2, lw=0.8)
a2.set_yticks(range(len(Q))); a2.set_yticklabels(Q.model, fontsize=7)
a2.set_xlabel("r(KEEP, correct) with 95% CI")
a2.set_title("b  Item sensitivity by tier", loc="left", fontsize=9, color=INK)
grid(a2, "x")
tier_legend(a2, ["Valid", "Indeterminate", "Invalid"], loc="lower right")
fig.savefig(FIG / "fig2_battery.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3 verdict depends on the cut point
A = pd.read_csv(OUT / "atlas" / "mmlu_per_model.csv")
B = pd.read_csv(OUT / "mmlu20" / "mmlu_per_model.csv")
Y = pd.read_csv(OUT / "yang" / "yang_basic_1s_cells.csv")
cuts = [50, 70, 80, 90, 95, 100]
panels = [("a  Atlas MMLU (33 models)", A, [f"c{c}_tier" for c in cuts]),
          ("b  MMLU-500 (20 models)", B, [f"c{c}_tier" for c in cuts]),
          ("c  Yang et al. (110 model-benchmark cells)", Y, [f"c{c}_tier" for c in cuts])]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True, gridspec_kw=dict(wspace=0.12))
for ax, (title, D, cols) in zip(axes, panels):
    for j, col in enumerate(cols):
        vc = D[col].value_counts(); n = vc.sum(); bottom = 0
        for t in ["Valid", "Indeterminate", "Invalid"]:
            v = vc.get(t, 0) / n
            if v > 0:
                ax.bar(j, v - 0.006, bottom=bottom + 0.003, width=0.72, color=TIER[t][0], ec="none")
            bottom += v
    ax.set_xticks(range(len(cuts))); ax.set_xticklabels([str(c) for c in cuts])
    ax.set_title(title, loc="left", fontsize=8, color=INK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.spines["left"].set_visible(ax is axes[0])
axes[0].set_ylabel("Share of units")
fig.supxlabel("Cut point c (confidence \u2265 c counts as high; 0\u20131 scale \u00d7 100 for Yang et al.)", fontsize=8, y=-0.04)
h = [plt.Rectangle((0, 0), 1, 1, color=TIER[t][0]) for t in ["Valid", "Indeterminate", "Invalid"]]
fig.legend(h, ["Valid", "Indeterminate", "Invalid"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98),
           fontsize=8, frameon=False)
fig.savefig(FIG / "fig3_cutpoints.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4 battery vs MMLU
X = pd.read_csv(OUT / "battery_vs_mmlu.csv")
fig, ax = plt.subplots(figsize=(4.8, 3.7))
for _, r in X.iterrows():
    col, mk = TIER[r.battery_tier]
    ax.plot([r.battery_r, r.battery_r], [r.atlas_lo, r.atlas_hi], color=col, lw=0.8, alpha=0.8, zorder=2)
    ax.plot(r.battery_r, r.atlas_auroc, marker=mk, mfc=col, mec=INK, mew=0.5, ms=6, ls="", zorder=3)
lab = {"DeepSeek-R1": (0.035, 0.035), "Gemini 3.1 Pro": (0.004, -0.075), "Qwen 80B Think": (0.04, 0.0),
       "Gemma 3 1B": (0.02, -0.03), "Gemma 3 12B": (0.02, -0.035), "Gemini 3 Flash": (0.03, 0.05),
       "GPT-5.4 nano": (-0.005, -0.06)}
for _, r in X[X.model.isin(lab)].iterrows():
    dx, dy = lab[r.model]
    ax.annotate(r.model, (r.battery_r, r.atlas_auroc), (r.battery_r + dx, r.atlas_auroc + dy), fontsize=7,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
ax.axhline(0.5, color=INK2, ls="--", lw=0.8)
ax.set_xlabel("Battery item sensitivity r(KEEP, correct)")
ax.set_ylabel("Atlas MMLU AUROC on raw confidence (95% CI)")
grid(ax)
h = [plt.Line2D([], [], ls="", marker=TIER[t][1], mfc=TIER[t][0], mec=INK, mew=0.5, ms=6, label=t)
     for t in ["Valid", "Indeterminate", "Invalid"]]
ax.legend(handles=h, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, title_fontsize=7.5,
          title=f"Battery tier  \u00b7  Spearman \u03c1 = {C['corr_battery_r_atlas_auroc']:+.2f}, n = 20")
fig.savefig(FIG / "fig4_battery_vs_mmlu.pdf"); plt.close(fig)
print("figures written to", FIG)
