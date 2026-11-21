#!/bin/bash
# Regenerate every number, table and figure in "Screen Before You Interpret" (BRM version)
# with validity-screen 0.4.1. Usage:
#   BATTERY=<battery csvs> ATLAS=<atlas data/raw (v2)> ATLAS_DATA=<atlas data/ (raw_v1, parsing counts)> \
#   MMLU500=<mmlu-500 csvs> \
#   YANG=<Yang et al. predictions/> HELDOUT=<saturation trials parquet> OUT=<out dir> PAPER=<paper dir> \
#   bash run_all.sh
set -euo pipefail
: "${BATTERY:?}" "${ATLAS:?}" "${ATLAS_DATA:?}" "${MMLU500:?}" "${YANG:?}" "${HELDOUT:?}" "${OUT:?}" "${PAPER:?}"
python3 -c "import validity_screen as v; assert v.__version__ == '0.4.1', v.__version__"
C="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT" "$PAPER/figures"
python3 -I "$C/screen_all.py" "$BATTERY" "$ATLAS" "$MMLU500" "$YANG" "$OUT" > "$OUT/screen_all.log"
python3 -I "$C/oc.py" "$OUT" "$BATTERY" "$ATLAS" > "$OUT/oc.log"
python3 -I "$C/atlas_exclusions.py" "$ATLAS_DATA" "$OUT" > "$OUT/atlas_exclusions.log"
python3 -I "$C/heldout.py" "$HELDOUT" "$OUT" step1 > "$OUT/heldout.log"
python3 -I "$C/heldout.py" "$HELDOUT" "$OUT" registered > "$OUT/heldout_registered.log"
python3 -I "$C/check_agreement.py" > "$OUT/check_agreement.log"
python3 -I "$C/paper_numbers.py" "$OUT" > "$OUT/paper_numbers.log"
python3 -I "$C/check_transfer.py" "$OUT" > "$OUT/check_transfer.log"
python3 -I "$C/tables.py" "$OUT" "$PAPER"
python3 -I "$C/figures.py" "$OUT" "$PAPER/figures"
echo "done: numbers in $OUT/numbers.json, oc_numbers.json, heldout_numbers.json"
