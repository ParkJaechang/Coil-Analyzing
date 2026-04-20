# Phase Residual <= 5 Hz Comparison

- scope: `<= 5 Hz interpolated continuous preview evaluation`

## Sine Subset

- baseline: `policy_eval_v2_upto5hz_sine_baseline.json`
- phase_residual: `policy_eval_v2_upto5hz_sine_phase_residual.json`
- case_count: `15`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| mean shape corr | 0.814311 | 0.814304 | -0.000007 |
| mean NRMSE | 0.235235 | 0.235240 | +0.000005 |
| mean phase lag (cycles) | -0.152604 | -0.159375 | -0.006771 |
| mean predicted error band | 0.220924 | 0.222823 | +0.001898 |
| false auto | 0 | 0 | +0 |
| false block | 1 | 1 | +0 |

## All <= 5 Hz Waveforms

- baseline: `policy_eval_v2_upto5hz_continuous_corpus_l1fo.json`
- phase_residual: `policy_eval_v2_upto5hz_phase_residual_all.json`
- case_count: `30`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| mean shape corr | 0.833503 | 0.833499 | -0.000003 |
| mean NRMSE | 0.200423 | 0.200425 | +0.000002 |
| mean phase lag (cycles) | -0.142318 | -0.145703 | -0.003385 |
| false auto | 0 | 0 | +0 |
| false block | 3 | 3 | +0 |

- exact_path_regression: `none observed in tests or exact export validation`
- interpolation_auto_reopen_judgement: `not supported by this result`
- conclusion: `phase-anchor residual model did not produce a meaningful <= 5 Hz interpolation gain; keep interpolated auto closed and shift focus back to exact-supported scope and finite data expansion.`