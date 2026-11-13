"""Count prospective-track 'errors' that are equivalent to the key (notation/format only).
Run: python3 -I prospective_regrade.py <csvdir> <outdir>
Rules are deliberately narrow: an answer is regraded correct only when it names the key's
referent in another notation (H2O/H₂O, ln|x| + C, (1+√5)/2, ...). Disputable keys are not touched.
"""
import glob, json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validity_screen import is_nonanswer  # noqa: E402

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
fs = sorted(f for f in glob.glob(str(IN / "*")) if "prosp" in f.lower())
x = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
x["answered"] = [not is_nonanswer(a, c) for a, c in zip(x.answer, x.correct_answer)]
w = x[x.answered & ~x.is_correct.astype(bool)].copy()
a = w.answer.astype(str)
RULES = {  # item: regex on the model's answer that names the key's referent
    "pr_000": r"H₂O|H2O", "pr_012": r"CO2|CO₂", "pr_024": r"5,?730", "pr_030": r"^\s*W\s*$",
    "pr_033": r"ln", "pr_039": r"5,?050", "pr_049": r"Beagle", "pr_050": r"Gibraltar",
    "pr_057": r"frac\{1\}\{12\}|1/12", "pr_059": r"[Cc]onsistent", "pr_063": r"e\s*[-−]\s*1",
    "pr_064": r"X[óo]õ|Taa", "pr_067": r"Zanzibar", "pr_069": r"√5|sqrt|[Gg]olden",
}
flip = pd.Series(False, index=w.index)
per_item = {}
for item, pat in RULES.items():
    m = (w.item_id == item) & a.str.contains(pat, regex=True)
    per_item[item] = int(m.sum())
    flip |= m
res = dict(scored_errors=int(len(w)), regraded_correct=int(flip.sum()),
           regraded_keep=int((w[flip].keep_withdraw.astype(str).str.upper() == "KEEP").sum()),
           per_item=per_item)
(OUT / "prospective_regrade.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
