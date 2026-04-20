# Exact Matrix Closeout

- generated_at_utc: `2026-04-16T02:43:41.605147+00:00`
- official_support_band_hz: `0.25 ~ 5.0`

## Continuous

- current_exact_auto: `{'status': 'operational', 'scope': 'continuous/current exact support only, <= 5 Hz', 'policy_version': 'v2'}`
- field_exact_status: `{'status': 'software_ready_bench_pending', 'scope': 'continuous/field exact support only, <= 5 Hz', 'remaining_gate': 'bench smoke test sign-off'}`
- interpolated_auto_enabled: `{'status': 'closed', 'scope': 'preview-only', 'reason': 'interpolated auto remains disabled'}`
- continuous_exact_grid_candidates: `[{'waveform': 'sine', 'freq_hz': 0.75, 'levels_a': [5.0, 10.0, 20.0]}, {'waveform': 'sine', 'freq_hz': 1.5, 'levels_a': [5.0, 10.0, 20.0]}, {'waveform': 'sine', 'freq_hz': 3.0, 'levels_a': [5.0, 10.0, 20.0]}, {'waveform': 'sine', 'freq_hz': 4.0, 'levels_a': [5.0, 10.0, 20.0]}, {'waveform': 'triangle', 'freq_hz': 0.75, 'levels_a': [5.0, 10.0, 20.0]}, {'waveform': 'triangle', 'freq_hz': 1.5, 'levels_a': [5.0, 10.0, 20.0]}]`

## Finite

- official_recipe_total: `95`
- sine_exact_total: `47`
- triangle_exact_total: `48`
- remaining_exact_gaps: `{'sine': [{'freq_hz': 1.0, 'cycles': 1.0, 'level_pp_a': 20}], 'triangle': []}`
- provisional_preview_combinations: `[{'waveform': 'sine', 'freq_hz': 1.0, 'cycles': 1.0, 'target_level_pp_a': 20, 'source_exact_level_pp_a': 10, 'scale_ratio': 2.0}]`

## UI State Validation

| case | support_state | preview_only | allow_auto_download | policy_version |
| --- | --- | --- | --- | --- |
| continuous_exact_current_auto | exact | False | True | v2 |
| continuous_exact_field_ready | exact | False | True | v2 |
| continuous_interpolated_preview | interpolated_in_hull | True | False | v2 |
| finite_provisional_sine_preview | provisional_preview | True | False | v2 |

## Request Router Validation

| case | clicked | freq_before | freq_after | target_after | finite_mode_after | cycle_after |
| --- | --- | --- | --- | --- | --- | --- |
| continuous_preview_apply_exact | 가장 가까운 exact 조합으로 전환 | 0.75 | 0.5 | 20.0 | False | None |
| finite_provisional_apply_exact | 가장 가까운 exact 조합으로 전환 | 1.0 | 1.0 | 10.0 | True | 1.0 |
| finite_provisional_apply_provisional | 가장 가까운 provisional 조합으로 미리보기 | 1.0 | 1.0 | 20.0 | True | 1.0 |

## Recommended Next Measurements

- `continuous sine exact grid: 0.75 Hz @ 5/10/20 A`
- `continuous sine exact grid: 1.5 Hz @ 5/10/20 A`
- `continuous sine exact grid: 3.0 Hz @ 5/10/20 A`
- `continuous sine exact grid: 4.0 Hz @ 5/10/20 A`

## Browser Export Validation

- scope: `default exact field path export rendering + selected file downloads`
- download_content_validation_passed: `True`
- clicked_download_buttons: `['제어 LUT CSV 다운로드', '보정 전압 파형 CSV 다운로드']`
- downloaded_files: `['compensated_voltage_waveform_sine_0.25Hz_field_20_steadycycle.csv', 'control_formula_sine_0.25Hz_field_20_control_lut.csv']`