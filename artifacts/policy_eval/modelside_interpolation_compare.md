# Model-side Interpolation Compare

- generated_at_utc: `2026-04-14T05:18:11.826135+00:00`

## Baseline

- case_count: `42`
- mean_shape_corr: `0.800618`
- mean_nrmse: `0.224879`
- mean_phase_lag_cycles: `-0.260045`

## Attempt: localbracket

- mean_shape_corr: `0.800614`
- mean_nrmse: `0.225882`
- mean_phase_lag_cycles: `-0.322731`
- mean_shape_corr_delta: `-0.000004`
- mean_nrmse_delta: `0.001003`
- mean_phase_lag_delta: `-0.062686`

### By Waveform

#### sine
- case_count: `21`
- mean_shape_corr_delta: `-0.000002`
- mean_nrmse_delta: `0.002003`
- mean_phase_lag_delta: `-0.059524`

#### triangle
- case_count: `21`
- mean_shape_corr_delta: `-0.000006`
- mean_nrmse_delta: `0.000003`
- mean_phase_lag_delta: `-0.065848`

## Attempt: phaseanchor

- mean_shape_corr: `0.800618`
- mean_nrmse: `0.225818`
- mean_phase_lag_cycles: `-0.312035`
- mean_shape_corr_delta: `0.000001`
- mean_nrmse_delta: `0.000939`
- mean_phase_lag_delta: `-0.051990`

### By Waveform

#### sine
- case_count: `21`
- mean_shape_corr_delta: `-0.000000`
- mean_nrmse_delta: `0.001882`
- mean_phase_lag_delta: `-0.105097`

#### triangle
- case_count: `21`
- mean_shape_corr_delta: `0.000002`
- mean_nrmse_delta: `-0.000004`
- mean_phase_lag_delta: `0.001116`

## Conclusion

- best_attempt_label: `phaseanchor`
- 이번 model-side interpolation 시도들은 평균 NRMSE 기준으로 baseline 대비 유의미한 개선을 만들지 못했습니다.