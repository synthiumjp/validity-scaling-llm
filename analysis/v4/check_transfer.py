"""Battery item sensitivity J (binary probe) vs atlas raw AUROC (verbalized), 19 models.
Run: python3 -I check_transfer.py <out_dir>"""
import sys
import numpy as np, pandas as pd
from scipy import stats
OUT = sys.argv[1]
C = pd.read_csv(f"{OUT}/cells.csv")
B = C[C.ds == "battery"].set_index("unit"); at = C[(C.ds == "atlas") & (C.cut == 90)].set_index("unit")
mp = {'Claude Haiku 4.5':'haiku-4-5','Opus 4.6':'opus-4-6','Sonnet 4.6':'sonnet-4-6','GPT-5.4':'gpt-5.4-2026','GPT-5.4 mini':'gpt-5.4-mini',
 'GPT-5.4 nano':'gpt-5.4-nano','Gemini 2.5 Flash':'gemini-2.5-flash','Gemini 2.5 Pro':'gemini-2.5-pro','Gemini 3 Flash':'gemini-3-flash',
 'Gemini 3.1 Pro':'gemini-3.1-pro','Gemma 3 1B':'gemma-3-1b','Gemma 3 12B':'gemma-3-12b','Gemma 3 27B':'gemma-3-27b','Qwen 235B':'qwen3-235b',
 'Qwen 80B Inst':'80b-a3b-instruct','Qwen 80B Think':'80b-a3b-thinking','Qwen Coder 480B':'coder-480b','DeepSeek V3.2':'deepseek-v3.2',
 'DeepSeek-R1':'deepseek-r1','GLM-5':'glm-5'}
rows = []
for b, key in mp.items():
    c = [u for u in at.index if key in u.lower()]
    assert len(c) == 1, (b, c)
    rows.append((b, B.J[b], at.auroc_raw[c[0]]))
D = pd.DataFrame(rows, columns=["model", "J", "atlas_auroc"]).dropna()
rho = stats.spearmanr(D.J, D.atlas_auroc)[0]
rng = np.random.default_rng(1); bs = []
for _ in range(5000):
    s = D.iloc[rng.integers(0, len(D), len(D))]
    if s.J.nunique() > 1:
        bs.append(stats.spearmanr(s.J, s.atlas_auroc)[0])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"n = {len(D)}, Spearman rho = {rho:.3f}, bootstrap 95% [{lo:.2f}, {hi:.2f}]")
