"""Run deterministic + stochastic synthetic policies; write synthetic_sim.json."""
import json, sys, runpy
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
OUT = Path(sys.argv[1])
ns = runpy.run_path(str(here / "test_synthetic.py"), run_name="synthetic")
rows, ok = ns["run"](verbose=False)
assert ok, "deterministic policy mismatch"
ns["test_identities"]()
res = {}
for ni in [40, 150]:
    res[str(ni)] = [dict(name=a, p_keep_correct=b, p_keep_incorrect=c, **d)
                    for a, b, c, d in ns["simulate"](n_correct=500 - ni, n_incorrect=ni, draws=2000)]
(OUT / "synthetic_sim.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
