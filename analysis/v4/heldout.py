"""Held-out application: the pre-registered saturation study (Cacioli, arXiv:2604.22215;
OSF https://osf.io/azbvx), rescored once with the frozen validity-screen 0.4 rule.

Pre-registered choices used unchanged (prereg v1.1, A2 and section 6.6):
  - confirmatory sample: the seven instruction-tuned models M2-M8;
  - conditions NUM (0-100 verbal) and CAT (10-class verbal);
  - scored rows: parse_status == 'success' (parse failures are reported, not scored);
  - cut point: parsed_confidence >= 0.50;
  - a cell with > 30% parse failure is reported but excluded from confirmatory counts.

Run: python3 -I heldout.py <trials.parquet> <out_dir> [step1|registered]
  step1 (primary): the screen's Step 1 also excludes class-only non-answers (a deviation
  from the registered scoring, which counted them as errors); registered: as pre-registered.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import validity_screen as vs

PQ, OUT = Path(sys.argv[1]), Path(sys.argv[2])
INSTRUCT = ["M2", "M3", "M4", "M5", "M6", "M7", "M8"]
CUT = 0.50
df = pd.read_parquet(PQ)

# Step 1 (non-answers). In the categorical format several models replied with a confidence
# class only (e.g. '"Highly likely" (0.8-0.9)'); the study's parser stored that class as the
# answer and the grader scored it wrong. A row is a non-answer if nothing substantive remains
# once the class label, its numeric range, quotes and filler wording are removed.
FILLER = set("i would classify my confidence in this answer as level is the for of category falls under "
             "a an it its rate be classified answer's my".split())


def class_only(answer, cls):
    if not isinstance(cls, str) or not cls.strip():
        return False
    a = str(answer).lower().replace(cls.lower().strip(), " ")
    a = re.sub(r"\(\s*[0-9.]+\s*[-\u2013]\s*[0-9.]+\s*\)", " ", a)
    words = [w for w in re.findall(r"[a-z0-9]+", a) if w not in FILLER]
    return len(words) == 0


df["class_only"] = [class_only(a, c) for a, c in zip(df["parsed_answer"], df["parsed_confidence_class"])]
SCORING = sys.argv[3] if len(sys.argv) > 3 else "step1"   # "step1" (primary) or "registered"
assert vs.__version__ == "0.4.1"

rows = []
for (m, cond), cell in df.groupby(["model_id", "condition"]):
    if m not in INSTRUCT:
        continue
    ok = (cell["parse_status"] == "success").values
    parse_rate = ok.mean()
    conf = pd.to_numeric(cell["parsed_confidence"], errors="coerce").values
    corr = cell["correct"].map(lambda x: 1.0 if x is True or x == 1 else (0.0 if x is False or x == 0 else np.nan)).values
    answered = ok & ~np.isnan(conf) & ~np.isnan(corr)
    n_class_only = int((ok & cell["class_only"].values).sum())
    if SCORING == "step1":
        answered &= ~cell["class_only"].values
    r = vs.screen(np.nan_to_num(corr), conf, answered=answered, cut=CUT, model_name=m,
                  benchmark_name="TriviaQA (524 items)", confidence_format=cond)
    s = answered
    au = (roc_auc_score(corr[s], conf[s]) if s.sum() and 0 < corr[s].mean() < 1 and np.std(conf[s]) > 0
          else np.nan)
    rows.append(dict(model=m, condition=cond, n_trials=len(cell), parse_rate=parse_rate,
                     n_class_only=n_class_only, scoring=SCORING,
                     confirmatory=parse_rate >= 0.70, n_scored=r.n_scored, n_err=r.n_incorrect,
                     accuracy=r.accuracy, share_ge95=float(np.mean(conf[s] >= 0.95)) if s.sum() else np.nan,
                     L=r.L.value, L_lo=r.L.lo, Fp=r.Fp.value, J=r.J.value, J_lo=r.J.lo, J_hi=r.J.hi,
                     p_signal=r.p_signal, auroc_raw=au, tier=r.tier, reason=r.reason_code,
                     flags="|".join(r.flags)))
H = pd.DataFrame(rows).sort_values(["condition", "model"])
OUT.mkdir(parents=True, exist_ok=True)
suffix = "" if SCORING == "step1" else "_registered"
H.to_csv(OUT / f"heldout_cells{suffix}.csv", index=False)
conf_num = H[(H.condition == "NUM") & H.confirmatory]
summary = dict(
    num_confirmatory_cells=int(len(conf_num)),
    num_tiers=conf_num.tier.value_counts().to_dict(),
    num_reasons=conf_num.reason.value_counts().to_dict(),
    cat_tiers=H[H.condition == "CAT"].tier.value_counts().to_dict(),
    cat_confirmatory=int(H[(H.condition == "CAT") & H.confirmatory].shape[0]),
)
(OUT / f"heldout_numbers{suffix}.json").write_text(json.dumps(summary, indent=1, default=float))
pd.set_option("display.width", 220)
print(H.round(3).to_string(index=False))
print(json.dumps(summary, indent=1, default=float))
