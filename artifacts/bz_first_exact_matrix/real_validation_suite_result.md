# Release Candidate Real Validation Suite

- 실제 validation 파일 기준으로 exact current / exact field / finite exact 3건을 end-to-end 재실행한 결과입니다.
- 각 케이스는 recommendation 또는 past export를 source로 선택하고 corrected LUT 산출물까지 생성합니다.

## continuous current exact real validation
- validation_source_file: `D:/programs/Codex/Coil Analyzing/.coil_analyzer/uploads/6c92e4f555df4fe7b6106580b286b88c_0.5Hz_9V_38gain.csv`
- exact_path: `exact_current`
- support_state: `exact`
- quality_label: `재현 양호`
- baseline_nrmse: `0.08987838796411285`
- corrected_nrmse: `0.05788248741926914`
- baseline_bz_nrmse: `0.07850690841520101`
- corrected_bz_nrmse: `0.051429118006393965`
- corrected_lut_id: `control_formula_sine_0.5Hz_current_20__corrected_iter01`
- report_path: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.5Hz_current_20__corrected_iter01_validation_report.json`

## continuous field exact real validation
- validation_source_file: `D:/programs/Codex/Coil Analyzing/.coil_analyzer/uploads/a45d8cb65a6b4e36b21481394aec8c0b_0.25Hz_9V_36gain.csv`
- exact_path: `exact_field`
- support_state: `exact`
- quality_label: `주의`
- baseline_nrmse: `0.1323377266881464`
- corrected_nrmse: `0.207454396593515`
- baseline_bz_nrmse: `0.1323377266881464`
- corrected_bz_nrmse: `0.20825717796436283`
- corrected_lut_id: `control_formula_sine_0.25Hz_field_20__corrected_iter01`
- report_path: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/control_formula_sine_0.25Hz_field_20__corrected_iter01_validation_report.json`

## finite exact real validation
- validation_source_file: `D:/programs/Codex/outputs/field_analysis_app_state/uploads/validation/8d51f7c793c4e3fa_샘플파형테스트결과_1hz_pp20A.csv`
- exact_path: `finite_exact`
- support_state: `exact`
- quality_label: `재보정 권장`
- baseline_nrmse: `0.35906447446159134`
- corrected_nrmse: `0.7500793862965061`
- baseline_bz_nrmse: `0.3523223298230427`
- corrected_bz_nrmse: `0.7420697596626517`
- corrected_lut_id: `ff132682eb37f728_1.25hz_1.25cycle_20pp__corrected_iter01`
- report_path: `D:/programs/Codex/Coil Analyzing/artifacts/validation_retune_real_example/ff132682eb37f728_1.25hz_1.25cycle_20pp__corrected_iter01_validation_report.json`
