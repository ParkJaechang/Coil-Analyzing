# Level Sensitivity Diagnosis

- comparisons: `6`
- support_id_switch: `1`
- prediction_source_switch: `1`
- limit_induced_switch: `0`
- true_nonlinear_shape_change: `0`

## finite_triangle_field_10 -> finite_triangle_field_20

- waveform/freq: `triangle / 1.0 Hz`
- level: `10.0` -> `20.0`
- switch_types: `prediction_source_switch`
- normalized_shape_difference: `4.645388209847426e-17`
- predicted_bz_shape_corr: `1.0`

## finite_triangle_field_20 -> finite_triangle_field_40

- waveform/freq: `triangle / 1.0 Hz`
- level: `20.0` -> `40.0`
- switch_types: `support_id_switch`
- normalized_shape_difference: `0.0`
- predicted_bz_shape_corr: `1.0`

## steady_state_harmonic_triangle_1.25Hz_field_40_1.25cycle -> control_formula_triangle_1.25Hz_field_40

- waveform/freq: `triangle / 1.25 Hz`
- level: `40.0` -> `40.0`
- switch_types: `none`
- normalized_shape_difference: `0.0`
- predicted_bz_shape_corr: `1.0`

## steady_state_harmonic_triangle_1.25Hz_field_20_1cycle -> control_formula_triangle_1.25Hz_field_20

- waveform/freq: `triangle / 1.25 Hz`
- level: `20.0` -> `20.0`
- switch_types: `none`
- normalized_shape_difference: `0.0`
- predicted_bz_shape_corr: `0.9999999999999999`

## steady_state_harmonic_triangle_0.5Hz_field_20_1cycle -> control_formula_triangle_0.5Hz_field_20

- waveform/freq: `triangle / 0.5 Hz`
- level: `20.0` -> `20.0`
- switch_types: `none`
- normalized_shape_difference: `0.0`
- predicted_bz_shape_corr: `1.0`

## steady_state_harmonic_triangle_1Hz_field_100_1.25cycle -> control_formula_triangle_1Hz_field_100

- waveform/freq: `triangle / 1.0 Hz`
- level: `100.0` -> `100.0`
- switch_types: `none`
- normalized_shape_difference: `0.0`
- predicted_bz_shape_corr: `1.0`
