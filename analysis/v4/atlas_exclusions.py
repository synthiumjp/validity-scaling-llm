"""Atlas Step 1 exclusion counts, derived from the released v1 corpus (data/raw_v1) and the
per-model parser route counts (data/parsing_fallback_counts.csv), using the same rule as the
atlas repository's scripts/v2_clean.py. Run: python3 -I atlas_exclusions.py <atlas data dir> <out_dir>"""
import json
import sys
from pathlib import Path

import pandas as pd

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
v1 = pd.concat([pd.read_csv(f) for f in sorted((DATA / "raw_v1").glob("*.csv"))]).drop_duplicates(["model", "item_id"])
v2 = pd.concat([pd.read_csv(f) for f in sorted((DATA / "raw").glob("*.csv"))])
fb = pd.read_csv(DATA / "parsing_fallback_counts.csv")
unparsed = v1.answer.astype(str).str.upper().eq("X")
imputed_models = set(fb.loc[fb.conf_imputed > 0, "model"])
bare = v1.model.str.split("/").str[-1].str.split("@").str[0]
is_imp_model = bare.map(lambda b: any(k == b or k.startswith(b + "-") or b.startswith(k) for k in imputed_models))
imputed = is_imp_model & v1.confidence.eq(50)
N = dict(v1_rows=len(v1), unparsed_answers=int(unparsed.sum()),
         parser_imputed_confidences=int(fb.conf_imputed.sum()),
         parser_imputed_largest_model=fb.loc[fb.conf_imputed.idxmax(), "model"],
         parser_imputed_largest_count=int(fb.conf_imputed.max()),
         removed_rows=int((unparsed | imputed).sum()), v2_rows=len(v2),
         consistent=len(v1) - int((unparsed | imputed).sum()) == len(v2))
(OUT / "atlas_exclusions.json").write_text(json.dumps(N, indent=1))
print(json.dumps(N, indent=1))
