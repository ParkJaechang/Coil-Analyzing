# Hardware Smoke Test Checklist

- generated_at_utc: `2026-04-14T03:06:41.954991+00:00`
- purpose: `exact-supported operational path confirmation only`
- note: `not executed in current coding environment; use on attached hardware bench`

## Continuous Exact Current

| case_id | waveform | freq_hz | target_current_pp_a | waveform_file | lut_file | shape_corr | nrmse | phase_lag_deg | clipping | pass_fail | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cont_exact_current_01 | sine | 0.5 | 20 | continuous_exact_current.csv | continuous_exact_current_control_lut.csv |  |  |  |  |  |  |

## Continuous Exact Field

| case_id | waveform | freq_hz | target_field_pp_mT | waveform_file | lut_file | shape_corr | nrmse | phase_lag_deg | clipping | pass_fail | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cont_exact_field_01 | sine | 0.25 | 20 | continuous_exact_field.csv | continuous_exact_field_control_lut.csv |  |  |  |  |  |  |

## Finite Exact Recipe

| case_id | waveform | freq_hz | cycles | level_pp_a | waveform_file | lut_file | shape_corr | nrmse | phase_lag_deg | clipping | pass_fail | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finite_exact_sine_01 | sine | 0.5 | 1.0 | 20 | finite_exact_sine.csv | finite_exact_sine_control_lut.csv |  |  |  |  |  |  |

## Acceptance

- continuous exact current: `shape corr >= 0.95`, `NRMSE <= 0.15`, no clipping
- continuous exact field: `shape corr >= 0.95`, `NRMSE <= 0.15`, no clipping
- finite exact recipe: `shape corr >= 0.90`, `NRMSE <= 0.20`, no clipping

## Artifact Files

- `cont_exact_current_01`
  waveform: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_current.csv`
  formula: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_current_formula.txt`
  lut: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_current_control_lut.csv`
- `cont_exact_field_01`
  waveform: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_field.csv`
  formula: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_field_formula.txt`
  lut: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_field_control_lut.csv`
- `finite_exact_sine_01`
  waveform: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\finite_exact_sine.csv`
  formula: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\finite_exact_sine_formula.txt`
  lut: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\finite_exact_sine_control_lut.csv`