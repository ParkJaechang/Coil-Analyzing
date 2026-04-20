# Validation Report

## Provenance
- exact_path: `exact_current`
- source_kind: `recommendation`
- lut_id: `control_formula_sine_0.5Hz_current_20`
- original_recommendation_id: `control_formula_sine_0.5Hz_current_20`
- corrected_lut_id: `control_formula_sine_0.5Hz_current_20__corrected_iter01`
- validation_run_id: `6c92e4f555df4fe7b6106580b286b88c_0.5Hz_9V_38gain::2026-04-17T16:34:48.455915`
- source_lut_filename: `control_formula_sine_0.5Hz_current_20.csv`
- iteration_index: `1`
- created_at: `2026-04-17T16:34:48.455915`
- correction_rule: `validation_residual_recommendation_loop[correction_gain=0.7;max_iterations=2;improvement_threshold=0]`

## Quality Badge
- label: `재현 양호`
- tone: `green`
- reasons: `Bz NRMSE / shape corr / phase lag가 exact retune 기준 내에 있습니다.`

## Quality Badge Rule
- metric domain: `bz_effective` (`bz_mT`, global rule `bz_effective = -bz_raw`)
- `재현 양호`: Bz NRMSE <= `0.15`, shape corr >= `0.97`, |phase lag| <= `0.02s`, clipping/saturation 없음
- `주의`: green 기준은 벗어나지만 Bz NRMSE <= `0.30`, shape corr >= `0.90`, |phase lag| <= `0.05s`, clipping/saturation 없음
- `재보정 권장`: clipping/saturation 감지 또는 Bz NRMSE > `0.30` 또는 shape corr < `0.90` 또는 |phase lag| > `0.05s`

## Before / After Metrics (Target Output Domain)
- before NRMSE: `0.0899`
- after NRMSE: `0.0579`
- before shape corr: `0.9941`
- after shape corr: `0.9994`
- before phase lag (s): `0.011758`
- after phase lag (s): `0.000000`

## Before / After Metrics (Bz Effective Domain)
- before Bz NRMSE: `0.0785`
- after Bz NRMSE: `0.0514`
- before Bz shape corr: `0.9956`
- after Bz shape corr: `0.9994`
- before Bz phase lag (s): `0.015677`
- after Bz phase lag (s): `0.000000`

## Retune Loop
- iteration_count: `2`
- stop_reason: `iteration_limit_reached`
- within_hardware_limits: `False`

## Artifacts
- corrected_waveform_csv: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.5Hz_current_20__corrected_iter01_waveform.csv`
- validation_report_json: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.5Hz_current_20__corrected_iter01_validation_report.json`
- validation_report_md: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.5Hz_current_20__corrected_iter01_validation_report.md`
- retune_result_json: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.5Hz_current_20__corrected_iter01_retune_result.json`
- retune_result_md: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.5Hz_current_20__corrected_iter01_retune_result.md`
- corrected_control_lut_csv: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.5Hz_current_20__corrected_iter01_control_lut.csv`
- corrected_formula_txt: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.5Hz_current_20__corrected_iter01_formula.txt`
