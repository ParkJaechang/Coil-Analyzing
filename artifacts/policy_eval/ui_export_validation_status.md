# UI Export Validation Status

- generated_at_utc: `2026-04-14T06:14:19.127910+00:00`
- goal: `operational path verification, not full browser automation completeness`

## Current/Exact Path

- status: `PASS`
- engine: `harmonic_surface`
- support_state: `exact`
- auto_state: `가능`
- source: `AppTest`

## Preview / Block Path

- preview_case_status: `PASS`
- preview_support_state: `interpolated_in_hull`
- finite_missing_exact_status: `PASS`
- source: `AppTest`

## Export Rendering

- status: `PASS`
- export_buttons_found: `5`
- export_button: `Harmonic Transfer LUT CSV 다운로드`
- export_button: `제어 수식 TXT 다운로드`
- export_button: `조화항 계수 CSV 다운로드`
- export_button: `제어 LUT CSV 다운로드`
- export_button: `보정 전압 파형 CSV 다운로드`
- source: `Selenium + headless Chrome`
- download_content_validation: `PASS`
- downloaded_file: `compensated_voltage_waveform_sine_0.25Hz_field_20_steadycycle.csv` size=`166274` header=`cycle_progress,time_s,target_output,used_target_output,target_field_mT,used_target_field_mT,recommended_voltage_v,recommended_voltage_pp,limited_voltage_v,limited_voltage_pp,max_daq_voltage_pp,max_daq_voltage_pk_v,peak_input_limit_margin,p95_input_limit_margin,required_amp_gain_multiplier,support_amp_gain_pct,required_amp_gain_pct,amp_gain_limit_pct,max_gain_pct_by_output,available_amp_gain_pct,amp_gain_at_100_pct,amp_max_output_pk_v,amp_output_pp_at_required,amp_output_pk_at_required,within_daq_limit,within_amp_gain_limit,within_amp_output_limit,within_hardware_limits,expected_current_a,expected_field_mT,expected_output,support_scaled_current_a,support_scaled_field_mT,modeled_current_a,modeled_field_mT,modeled_output,aligned_target_output,aligned_used_target_output,aligned_target_field_mT,aligned_used_target_field_mT,target_output_pp,waveform_type,freq_hz,finite_cycle_mode,target_cycle_count,preview_tail_cycles,estimated_output_lag_seconds,estimated_output_lag_cycles`
- downloaded_file: `control_formula_sine_0.25Hz_field_20_control_lut.csv` size=`8526` header=`lut_index,time_s,command_voltage_v,finite_cycle_mode,cycle_progress`

## Limits

- current-target browser rerender was not stabilized in this headless route
- current-target operational state is still covered by AppTest evidence
- browser validation confirms export section/button rendering and basic downloaded file content
