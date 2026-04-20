# Measurement ROI Priority

- The list below ranks the next measurement or validation actions by operational value.
- Priority 1 changes dynamically with the current matrix state instead of staying pinned to the same request.
- Current head category: missing_exact_promotion.
- Continuous exact gap-fill and validation actions remain separate so operators can plan bench time deliberately.

## Priority Queue

| rank | category | request | why | expected_gain |
| --- | --- | --- | --- | --- |
| 1 | missing_exact_promotion | current / sine / 1 Hz / 1 cycle / 20 pp | Only remaining certified finite exact gap. A measured upload promotes the matrix from 95 to 96 certified exact cells. | removes the last provisional-only cell from operator guidance |
| 2 | continuous_exact_gap_fill | current / 0.75 Hz | waveforms sine, triangle | levels 5 A, 10 A, 20 A | fills the gap between 0.5 Hz and 1.0 Hz | reduces preview-only interpolation in the continuous exact operating table |
| 3 | continuous_exact_gap_fill | current / 1.5 Hz | waveforms sine, triangle | levels 5 A, 10 A, 20 A | fills the gap between 1.0 Hz and 2.0 Hz | reduces preview-only interpolation in the continuous exact operating table |
| 4 | continuous_exact_gap_fill | current / 3 Hz | waveforms sine, triangle | levels 5 A, 10 A, 20 A | reduces the long jump between 2.0 Hz and 5.0 Hz | reduces preview-only interpolation in the continuous exact operating table |
| 5 | continuous_exact_gap_fill | current / 4 Hz | waveforms sine, triangle | levels 5 A, 10 A, 20 A | adds a pre-5 Hz anchor close to the exact operating limit | reduces preview-only interpolation in the continuous exact operating table |
| 6 | validation_priority | current / sine / 1.25 Hz / 1.25 cycle / 20 pp | export | validate 1.25 cycle | clipping/saturation detected; Bz NRMSE 74.21%; Bz shape corr 0.368; Bz phase lag 0.1498s | converts existing LUT lineage into a trustworthy operator reference |
| 7 | finite_edge_reinforcement | current / finite edge reinforcement | freqs 2 Hz, 5 Hz | cycles 0.75 cycle, 1 cycle, 1.25 cycle, 1.5 cycle | waveforms sine, triangle | level 20 pp | 2 Hz and 5 Hz are the highest-stress finite exact bands and worth reinforcing with repeat measurements. | improves confidence near the dynamic edge of the finite exact matrix |
