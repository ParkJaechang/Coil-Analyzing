# Validation / Retune Report

## Run Linkage
- LUT ID: `mock_current_exact_retune`
- target type: `current`
- waveform: `sine`
- freq_hz: `1.0`
- cycles: `None`
- target level: `10.0` (pp)
- validation test: `mock_validation_current_1hz`
- measured file: `n/a`

## Baseline vs Corrected
- baseline NRMSE: `0.2374`
- corrected NRMSE: `0.0149`
- baseline shape corr: `0.9510`
- corrected shape corr: `0.9999`
- baseline phase lag (s): `0.047712`
- corrected phase lag (s): `0.000000`
- baseline pp error: `-1.8000`
- corrected pp error: `-0.1370`

## Retune Loop
- iteration_count: `2`
- stop_reason: `iteration_limit_reached`
- within_hardware_limits: `True`
- correction_gain: `0.75`
- improvement_threshold: `0.0`
