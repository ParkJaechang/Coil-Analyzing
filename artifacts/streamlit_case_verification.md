# Streamlit Case Verification

- verification_mode: `streamlit_apptest_ui_execution`
- live_browser_note: ???? ??? ?? artifact? ?????, ? ????? Streamlit form submit Selenium ???? ????? ?? per-case ?? AppTest? ????.

## continuous_exact_current_0p5hz
- request_route: `exact`
- solver_route: `harmonic_surface_inverse_exact`
- plot_source: `exact_prediction`
- support_state: `exact`
- allow_auto_download: `False`
- selected_support_waveform: `sine`
- harmonic_weights_used: `{1: 1.0}`
- predicted_shape_corr: `0.884`
- predicted_nrmse: `0.339`
- predicted_phase_lag: `+0.141176s`
- auto_gate_reasons: `predicted_shape_corr_below_threshold,predicted_nrmse_above_threshold`
- plots:
  - Current Waveform Compensation: ['Target Output', 'Exact Predicted Output']
  - Recommended Command Waveform: ['limited_voltage_v']

## continuous_preview_interpolated_0p75hz
- request_route: `preview`
- solver_route: `harmonic_surface_inverse_interpolated_preview`
- plot_source: `support_blended_preview`
- support_state: `interpolated_in_hull`
- allow_auto_download: `False`
- selected_support_waveform: `sine`
- harmonic_weights_used: `{1: 1.0}`
- predicted_shape_corr: `0.794`
- predicted_nrmse: `0.454`
- predicted_phase_lag: `+0.125490s`
- auto_gate_reasons: `predicted_error_band_above_threshold,input_limit_margin_below_threshold,surface_confidence_below_threshold,predicted_shape_corr_below_threshold,predicted_nrmse_above_threshold`
- plots:
  - Current Waveform Compensation: ['Target Output', 'Predicted Output', 'Support-Blended Output']
  - Recommended Command Waveform: ['limited_voltage_v']

## finite_triangle_exact_1hz_1cycle_20pp
- request_route: `exact`
- solver_route: `finite_exact_direct`
- plot_source: `exact_prediction`
- support_state: `exact`
- allow_auto_download: `False`
- selected_support_waveform: `triangle`
- harmonic_weights_used: `{1: 1.0, 3: 2.4, 5: 1.8, 7: 1.4}`
- predicted_shape_corr: `0.773`
- predicted_nrmse: `1.630`
- predicted_phase_lag: `+0.102941s`
- auto_gate_reasons: `predicted_shape_corr_below_threshold,predicted_nrmse_above_threshold,predicted_phase_lag_above_threshold`
- plots:
  - Current Waveform Compensation: ['Target Output', 'Exact Predicted Output']
  - Recommended Command Waveform: ['limited_voltage_v']
  - Recommended Command Waveform: ['limited_voltage_v']

## finite_provisional_sine_1hz_1cycle_20pp
- request_route: `exact`
- solver_route: `finite_exact_direct`
- plot_source: `exact_prediction`
- support_state: `provisional_preview`
- allow_auto_download: `False`
- selected_support_waveform: `sine`
- harmonic_weights_used: `{1: 1.0, 3: 0.2, 5: 0.08, 7: 0.03}`
- predicted_shape_corr: `0.767`
- predicted_nrmse: `1.657`
- predicted_phase_lag: `+0.112745s`
- auto_gate_reasons: `predicted_shape_corr_below_threshold,predicted_nrmse_above_threshold,predicted_phase_lag_above_threshold`
- plots:
  - Current Waveform Compensation: ['Target Output', 'Exact Predicted Output']
  - Recommended Command Waveform: ['limited_voltage_v']
  - Recommended Command Waveform: ['limited_voltage_v']

## finite_unsupported_triangle_6hz_1p25cycle_20pp
- request_route: `exact`
- solver_route: `finite_exact_direct`
- plot_source: `exact_prediction`
- support_state: `unsupported`
- allow_auto_download: `False`
- selected_support_waveform: `triangle`
- harmonic_weights_used: `{1: 1.0, 3: 2.4, 5: 1.8, 7: 1.4}`
- predicted_shape_corr: `0.172`
- predicted_nrmse: `1.661`
- predicted_phase_lag: `+0.034314s`
- auto_gate_reasons: `legacy_engine_used`
- plots:
  - Current Waveform Compensation: ['Target Output', 'Exact Predicted Output']
  - Recommended Command Waveform: ['limited_voltage_v']
  - Recommended Command Waveform: ['limited_voltage_v']

## FFT Compare
- continuous_exact_sine_odd_energy: `3.7909818394774595`
- continuous_exact_triangle_odd_energy: `17.361833598455725`
- continuous_triangle_to_sine_odd_ratio: `4.579772294780722`