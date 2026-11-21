# Analysis with validity-screen 0.4.1

This pipeline rescreens every dataset in the programme with the 0.4 decision rules and computes the screen's operating characteristics. It is the analysis behind the next version of the paper, which is in preparation. The published paper (arXiv:2604.17714v2) used 0.3.0 and is reproduced by `analysis/v2/`.

The rules and the reasons for each change are in [`screen/DECISIONS.md`](../../screen/DECISIONS.md).

## What changed from 0.3.0 to 0.4

- **Invalid means a fixed response set, judged the same way in both directions:** blanket confidence (kept-error rate L ≥ .95) or blanket withdrawal (withdrawn-correct rate Fp ≥ .95), each certified by a Wilson lower bound above .90. A second route to Invalid is a certified inversion. Under 0.3.0, Fp ≥ .50 alone made a model Invalid.
- **Fp ≥ .50 is now a low-coverage flag**, reported beside the tier.
- **Valid needs evidence of signal:** a one-sided Fisher mid-p test of J > 0 at .025.
- **Every Indeterminate carries a reason code.**
- **Continuous confidence has no default cut point.**

## Results under 0.4.1

**Battery** (20 models, binary KEEP/WITHDRAW probe, 452 items):
- 15 Valid and 4 Indeterminate: 2 with no detectable sensitivity, 2 with blanket confidence suspected but not certified.
- 1 Uncertifiable: DeepSeek-R1, which makes 6 errors and carries the low-coverage flag. It was Invalid under 0.3.0.

**Verbalized 0–100 confidence.** Valid counts by cut point:

| Dataset | 50 | 70 | 80 | 90 | 95 | 100 |
|---|---|---|---|---|---|---|
| Atlas (33 models, 46,157 scored responses) | 9 | 13 | 18 | 28 | 31 | 25 |
| MMLU-500 (20 models) | 4 | 7 | 9 | 13 | 16 | 10 |

At low cut points, most models are Invalid for blanket confidence. No cut point is primary: these are descriptive.

**Held-out saturation study** (arXiv:2604.22215, OSF azbvx), rescored once with the frozen rule:
- In the 0–100 condition, 6 of 7 confirmatory cells are Invalid (blanket confidence) and 1 is Valid.
- The primary scoring excludes class-only non-answers at Step 1. This is a deviation from the registered scoring, which counted them as errors. Both scorings are produced.

**Operating characteristics (exact enumeration)** — false-Valid rate under no signal, against a nominal .025:

| Case | Worst false-Valid rate |
|---|---|
| Typical cases (226–1,500 items, accuracy .70–.97, keep .50–.99) | .026 |
| Only 10–12 errors, keep ≥ .50 | .030 |
| Low keep rates (.10–.30) | .037 |

False Invalid: a model whose true kept-error rate is .93 is certified as blanket confidence up to 23% of the time.

**Implementation check.** The package agrees with an independent reimplementation, written from the text alone, on all 130,320 tables tested (`out/check_agreement.log`).

## Running it

```bash
pip install -r analysis/v4/requirements.txt validity-screen==0.4.1
BATTERY=data/csvs \
ATLAS=<metacognitive-profile-atlas>/data/raw \
ATLAS_DATA=<metacognitive-profile-atlas>/data \
MMLU500=cross-benchmark/mmlu/item_level \
YANG=<Yang et al. predictions>/ \
HELDOUT=<saturation study trials.parquet> \
OUT=analysis/v4/out PAPER=<paper dir> \
bash analysis/v4/run_all.sh
```

### Data sources

- The atlas data are at [synthiumjp/metacognitive-profile-atlas](https://github.com/synthiumjp/metacognitive-profile-atlas).
- The held-out trials are from the saturation study's release.
- Yang et al.'s predictions are released by the original authors (arXiv:2412.14737) and are not redistributed here.

### Files

| Script | Produces |
|---|---|
| `screen_all.py` | `out/cells.csv`: every model × dataset × cut, with 0.4.1 and 0.3.0 tiers |
| `oc.py` | `out/oc_*.csv`, `oc_numbers.json`: size, false Invalid, power, items needed, multiplicity, two-run agreement, binarisation loss |
| `heldout.py` | `out/heldout_*`: the held-out rescoring, primary (`step1`) and as registered |
| `atlas_exclusions.py` | `out/atlas_exclusions.json`: Step 1 exclusion counts for the atlas |
| `check_agreement.py` | agreement with the independent implementation in `independent/` |
| `check_transfer.py`, `paper_numbers.py`, `tables.py`, `figures.py` | summary numbers, LaTeX tables, figures |
| `legacy/` | the frozen 0.3.0 rule, for the comparison table |
| `exploratory/` | development-data feasibility checks for the planned predictive-validity analysis; not part of `run_all.sh` |
