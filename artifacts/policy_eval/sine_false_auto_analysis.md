# Sine False Auto Analysis

- input: `artifacts\policy_eval\policy_eval_v3_candidate_p95_continuous_corpus_l1fo.csv`
- false_auto_total: `17`
- sine_false_auto_total: `17`
- triangle_false_auto_total: `0`

## Common Pattern

- All false auto cases are sine steady-state interpolation cases.
- For sine false auto, surface_confidence is constant at 0.6 and predicted_error_band is constant at 0.16, so the policy cannot discriminate within that slice.
- Harmonic fill ratio is saturated at 1.0 on every sine false auto case, so harmonic coverage is not the differentiating factor.
- Predicted error calibration is structurally weak: on the full corpus, predicted_error_band shows weak inverse correlation with realized_nrmse, while surface_confidence shows weak positive correlation with realized_nrmse.
- Actual interpolation quality degrades strongly with frequency for sine (especially 2 Hz and 5 Hz), but the current confidence model does not include frequency-distance or bracket-span penalties.

## Constant Policy Inputs On Sine False Auto

- `surface_confidence`: `[0.6000000000000001]`
- `predicted_error_band`: `[0.1599999999999999]`
- `harmonic_fill_ratio`: `[1.0]`
- `support_run_count`: `[23, 24]`

## Frequency Error Summary

| target_freq_hz | realized_nrmse_mean | realized_nrmse_max | shape_corr_mean | shape_corr_max | current_pp_error_pct_mean | current_pp_error_pct_max | phase_lag_deg_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.126121 | 0.128792 | 0.945108 | 0.947598 | 6.373246 | 7.599691 | -56.601562 |
| 1.0 | 0.235856 | 0.239566 | 0.851266 | 0.895577 | 20.997611 | 31.39899 | -60.46875 |
| 2.0 | 0.346608 | 0.40521 | 0.642455 | 0.782517 | 26.155961 | 63.323084 | -66.9375 |
| 5.0 | 0.379029 | 0.419233 | 0.599169 | 0.632184 | 32.45441 | 46.303877 | -240.0 |

## Target Level Summary

| target_level_bin | case_count | realized_nrmse_mean | shape_corr_mean | current_pp_error_pct_mean | phase_lag_deg_mean |
| --- | --- | --- | --- | --- | --- |
| (-0.001, 1.0] | 5 | 0.366042 | 0.581209 | 26.443176 | -163.96875 |
| (1.0, 3.0] | 2 | 0.262559 | 0.760196 | 14.590522 | -38.671875 |
| (3.0, 6.0] | 6 | 0.263972 | 0.814593 | 24.631916 | -75.9375 |
| (6.0, 20.0] | 4 | 0.153684 | 0.933139 | 12.358914 | -57.65625 |

## Support Density Summary

| support_run_count | case_count | realized_nrmse_mean | shape_corr_mean | current_pp_error_pct_mean |
| --- | --- | --- | --- | --- |
| 23 | 14 | 0.244058 | 0.803503 | 18.661489 |
| 24 | 3 | 0.379029 | 0.599169 | 32.45441 |

## Interpolation Geometry Summary

| freq_hz | nearest_abs_hz_mean | lower_hz_mean | upper_hz_mean | bracket_span_hz_mean | bracket_position_mean |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 0.25 | 0.25 | 1.0 | 0.75 | 0.333333 |
| 1.0 | 0.5 | 0.5 | 2.0 | 1.5 | 0.333333 |
| 2.0 | 1.0 | 1.0 | 5.0 | 4.0 | 0.25 |
| 5.0 | 3.0 | 2.0 | 10.0 | 8.0 | 0.375 |

## One-Line Conclusion

- This is not a threshold-only problem; it is primarily a confidence-design problem, with steady-state harmonic-surface interpolation overestimating trust because it ignores interpolation geometry.

## Shortest Next Fix

- Add interpolation-geometry-aware penalties to surface_confidence and predicted_error_band for steady-state sine interpolation, using nearest support distance and bracketing span in log-frequency space.