# Feature Progress Report

## Interpolation Confidence L1FO Comparison

### v2 (operational)
- policy_version: `v2_continuous_corpus_l1fo`
- auto_count: `0`
- false_auto_count: `0`
- false_block_count: `3`
- mean_realized_shape_corr: `0.800618`
- mean_realized_nrmse: `0.224879`
- mean_predicted_error_band: `0.164816`

### v3_candidate_p95
- policy_version: `v3_candidate_p95_continuous_corpus_l1fo`
- auto_count: `18`
- false_auto_count: `17`
- false_block_count: `2`
- mean_realized_shape_corr: `0.800618`
- mean_realized_nrmse: `0.224879`
- mean_predicted_error_band: `0.164816`

### v3_geom_p95
- policy_version: `v3_geom_p95_continuous_corpus_l1fo`
- auto_count: `0`
- false_auto_count: `0`
- false_block_count: `3`
- mean_realized_shape_corr: `0.800618`
- mean_realized_nrmse: `0.224879`
- mean_predicted_error_band: `0.190450`

## Exact Export Validation

### continuous_exact_current
- support_state: `exact`
- preview_only: `False`
- allow_auto_download: `True`
- csv_size_bytes: `165804`
- formula_size_bytes: `564`
- lut_size_bytes: `8649`
- line_count: `257`
- coeff_row_count: `10`
- lut_row_count: `128`

### continuous_exact_field
- support_state: `exact`
- preview_only: `False`
- allow_auto_download: `True`
- csv_size_bytes: `168757`
- formula_size_bytes: `570`
- lut_size_bytes: `8645`
- line_count: `257`
- coeff_row_count: `10`
- lut_row_count: `128`

### finite_exact_sine
- support_state: `exact`
- preview_only: `False`
- allow_auto_download: `True`
- csv_size_bytes: `128531`
- formula_size_bytes: `618`
- lut_size_bytes: `8338`
- line_count: `257`
- coeff_row_count: `10`
- lut_row_count: `128`

## UI Policy Validation

### exact_current_auto
- pass_status: `True`
- warnings: `1`
- successes: `3`
- engine: `harmonic_surface`
- support_state: `exact`
- auto_recommendation: `가능`

### exact_field_auto
- pass_status: `True`
- warnings: `1`
- successes: `2`
- engine: `harmonic_surface`
- support_state: `exact`
- auto_recommendation: `가능`

### interpolated_current_preview
- pass_status: `True`
- warnings: `2`
- successes: `3`
- engine: `harmonic_surface`
- support_state: `interpolated_in_hull`
- auto_recommendation: `preview-only`

### finite_exact_supported
- pass_status: `True`
- warnings: `3`
- successes: `3`
- engine: `legacy`
- support_state: `exact`
- auto_recommendation: `가능`

### finite_missing_exact_recipe
- pass_status: `True`
- warnings: `3`
- successes: `3`
- engine: `legacy`
- support_state: `exact`
- auto_recommendation: `preview-only`

## Finite Generalization Stage 2

- case_count: `47`
- preview_case_count: `47`
- mean_shape_corr: `0.157136`
- mean_nrmse: `0.384775`
- mean_phase_lag_s: `-0.180646`

## Current Status

- continuous/current exact path: validated and unchanged
- continuous interpolation: geometry-aware confidence added, but operational rollout remains closed
- exact field path: export + UI policy path validated
- finite exact path: export + UI policy path validated
- finite preview stage 2: quantified, preview-only quality remains weak on corpus LORO
- hardware smoke test: not executed in this environment
