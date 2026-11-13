#!/bin/bash
# Regenerate every number, table and figure in arXiv:2604.17714v2.
#
# Run from the repository root:
#   ATLAS_DIR=/path/to/metacognitive-profile-atlas/data/raw \
#   YANG_DIR=/path/to/yang2024/predictions \
#   bash analysis/v2/run_all.sh
#
# ATLAS_DIR : the 33 item-level CSVs from github.com/synthiumjp/metacognitive-profile-atlas (data/raw)
# YANG_DIR  : the `predictions/` folder (y_pred/, y_true/) released by Yang, Tsai and Yamada (2024),
#             arXiv:2412.14737. Not redistributed here.
# Requires: python >= 3.9, numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=analysis/v2
O=$C/out                    # primary: five keyed battery tracks
O6=$C/out_sixtrack          # sensitivity: all six tracks, non-answers excluded
B=data/csvs
M=cross-benchmark/mmlu/item_level
: "${ATLAS_DIR:?set ATLAS_DIR to the atlas data/raw folder}"
: "${YANG_DIR:?set YANG_DIR to the Yang et al. predictions folder}"
mkdir -p $O/atlas $O/mmlu20 $O/yang $O/yang_gt0 $O6 paper/v2/figures

VS_DROP_TRACKS=prospective python -I $C/rescore.py $B $O > /dev/null
python -I $C/rescore.py $B $O6 > /dev/null
VS_DROP_TRACKS=prospective python -I $C/extra_17707.py $B $O > /dev/null
python -I $C/mmlu_rescore.py "$ATLAS_DIR" $O/atlas '{"gpt-oss-120b": 50}' > $O/atlas/log.txt
python -I $C/mmlu_rescore.py $M $O/mmlu20 > $O/mmlu20/log.txt
python -I $C/yang_rescore.py "$YANG_DIR" $O/yang 0.5 > $O/yang/log.txt
python -I $C/yang_rescore.py "$YANG_DIR" $O/yang_gt0 0.000001 > $O/yang_gt0/log.txt
python -I $C/synthetic_run.py $O > /dev/null
python -I $C/prospective_regrade.py $B $O > /dev/null
python -I $C/cross_benchmark.py $O > /dev/null
python -I $C/tables_v2.py $O paper/v2 > /dev/null
python -I $C/figures_v2.py $O paper/v2/figures > /dev/null
echo "Done: outputs in $O and $O6; tables and figures in paper/v2."
