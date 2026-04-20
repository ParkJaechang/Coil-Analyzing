# Model-side Interpolation Attempts

- generated_at_utc: `2026-04-14T05:18:10.203931+00:00`

## baseline_v2

- policy_version: `v2_continuous_corpus_l1fo`
- case_count: `42`
- auto_count: `0`
- false_auto_count: `0`
- false_block_count: `3`
- mean_shape_corr: `0.800618`
- mean_nrmse: `0.224879`
- mean_predicted_error_band: `0.164816`

## localbracket_attempt

- policy_version: `v2_modelside_localbracket_continuous_corpus_l1fo`
- case_count: `42`
- auto_count: `0`
- false_auto_count: `0`
- false_block_count: `3`
- mean_shape_corr: `0.800614`
- mean_nrmse: `0.225882`
- mean_predicted_error_band: `0.190450`

## phaseanchor_attempt

- policy_version: `v2_modelside_phaseanchor_continuous_corpus_l1fo`
- case_count: `42`
- auto_count: `0`
- false_auto_count: `0`
- false_block_count: `3`
- mean_shape_corr: `0.800618`
- mean_nrmse: `0.225818`
- mean_predicted_error_band: `0.190450`

## Conclusion

- 이번 model-side interpolation 2회 시도는 baseline 대비 유의미한 quality 개선을 만들지 못했습니다.
- exact path regression: `none observed in exact export validation`
- operational decision: `interpolated auto remains closed`