# Finite Data Recommendations

- generated_at_utc: `2026-04-17T07:11:55.222022+00:00`
- official_support_band_hz: `0.25 ~ 5.0`

## Stage-2 Preview Summary

- mean_shape_corr: `0.15728803598624605`
- mean_nrmse: `0.4320686361487456`
- mean_phase_lag_s: `-0.18064553141772746`

- missing_exact_combinations: `[]`

## Recommended Next Measurements

- P1: Measure the missing exact finite recipe: 1.0 Hz + 1.0 cycle + 20 pp. (This is the only missing combination inside the current <= 5 Hz exact recipe table.)
- P2: Add exact finite transient data at 2.0 Hz and 5.0 Hz. (Stage-2 preview quality remains weakest in the upper part of the official <= 5 Hz band.)
- P3: Repeat 0.75 cycle and 1.5 cycle measurements at 1.0 Hz and 1.25 Hz. (Mid-band phase lag and preview consistency are still unstable across repeated finite runs.)

- conclusion: `Finite should remain exact-recipe-first. The fastest improvement is to close the one missing exact recipe and strengthen 2/5 Hz transient coverage before revisiting any generalization work.`