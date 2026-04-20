# Hardware Smoke Test Results Template

- generated_at_utc: `2026-04-14T05:18:10.396947+00:00`
- source_manifest: `artifacts\policy_eval\hardware_smoke_test_manifest.json`

## Recording Rules

- shape_corr / nrmse / phase_lag_s는 실측 파형과 목표 파형 비교 기준으로 기록합니다.
- clipping_or_saturation은 `true` 또는 `false`로 기록합니다.
- operator_judgement는 `목표 개형 재현 가능` 또는 `재현 불가`로 기록합니다.
- pass_fail은 `PASS` 또는 `FAIL`로 기록합니다.

## Cases

### cont_exact_current_01
- category: `continuous_current`
- description: `exact auto`
- regime: `continuous`
- target_type: `current`
- waveform_type: `sine`
- freq_hz: `0.5`
- level: `20`
- cycle_count: `None`
- lut_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_current_control_lut.csv`
- formula_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_current_formula.txt`
- waveform_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_current.csv`
- shape_corr: ``
- nrmse: ``
- phase_lag_s: ``
- clipping_or_saturation: ``
- operator_judgement: ``
- pass_fail: ``
- notes: ``

### cont_exact_field_01
- category: `continuous_field`
- description: `exact auto`
- regime: `continuous`
- target_type: `field`
- waveform_type: `sine`
- freq_hz: `0.25`
- level: `20`
- cycle_count: `None`
- lut_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_field_control_lut.csv`
- formula_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_field_formula.txt`
- waveform_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\continuous_exact_field.csv`
- shape_corr: ``
- nrmse: ``
- phase_lag_s: ``
- clipping_or_saturation: ``
- operator_judgement: ``
- pass_fail: ``
- notes: ``

### finite_exact_sine_01
- category: `transient_current`
- description: `exact recipe`
- regime: `transient`
- target_type: `current`
- waveform_type: `sine`
- freq_hz: `0.5`
- level: `20`
- cycle_count: `1.0`
- lut_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\finite_exact_sine_control_lut.csv`
- formula_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\finite_exact_sine_formula.txt`
- waveform_file: `D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\export_validation\finite_exact_sine.csv`
- shape_corr: ``
- nrmse: ``
- phase_lag_s: ``
- clipping_or_saturation: ``
- operator_judgement: ``
- pass_fail: ``
- notes: ``
