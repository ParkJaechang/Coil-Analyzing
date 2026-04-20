# LCR Prior <= 5 Hz Sine Comparison

- baseline: `policy_eval_v2_upto5hz_sine_baseline.json`
- lcr: `policy_eval_v2_upto5hz_sine_lcrprior035.json`
- case_count: `15`

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| mean shape corr | 0.814311 | 0.814311 | -0.000000 |
| mean NRMSE | 0.235235 | 0.235235 | -0.000000 |
| mean predicted error band | 0.220924 | 0.229893 | +0.008969 |
| false auto | 0 | 0 | +0 |
| false block | 1 | 1 | +0 |

- conclusion: `LCR prior infrastructure is active, but the current 0.35 blend does not materially improve <=5 Hz sine L1FO preview quality yet; the next shortest change should target stronger harmonic-local phase modeling rather than policy reopening.`