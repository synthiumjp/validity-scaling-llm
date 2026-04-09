# Cross-Benchmark Validation

Two independent cross-benchmark validations testing whether the validity screening protocol (Cacioli, 2026e) transfers beyond the derivation battery.

## 1. Yang et al. (2024) — External dataset

11 models across 10 benchmarks with verbalized confidence (0-100). One model (Qwen 1.5-7B-chat) classified Invalid. All 10 others Valid. Spearman rho(L, AUROC) = .894, p < .000001.

See [`yang2024/`](yang2024/) for details.

## 2. MMLU — Same models, different benchmark and probe format

18 of 20 battery models evaluated on 500 stratified MMLU items with verbalized confidence (0-100). All battery-Valid models remained Valid. All three battery-Invalid models shifted to Valid under the continuous probe format. Gemma 1B remained Invalid on both formats.

See [`mmlu/`](mmlu/) for details.

## Key conclusion

The screen transfers across benchmarks and independent datasets. Probe format modulates classification: binary KEEP/WITHDRAW is the harder test. A model that passes the binary screen will pass the continuous screen. The reverse is not true.

## Tool

```bash
pip install validity-screen
```

## Citation

- Cacioli, J. P. (2026e). Screen Before You Interpret. arXiv.
- Yang, S., et al. (2024). arXiv:2404.09272 (external dataset).
