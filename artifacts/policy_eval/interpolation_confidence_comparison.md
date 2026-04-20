# Interpolation Confidence Comparison

| label | policy_version | case_count | auto_count | false_auto | false_block | correct_auto | mean_shape_corr | mean_nrmse | mean_predicted_error_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_operational | v2_continuous_corpus_l1fo | 42 | 0 | 0 | 3 | 0 | 0.800618 | 0.224879 | 0.164816 |
| v3_candidate_p95 | v3_candidate_p95_continuous_corpus_l1fo | 42 | 18 | 17 | 2 | 1 | 0.800618 | 0.224879 | 0.164816 |
| v3_geom_p95 | v3_geom_p95_continuous_corpus_l1fo | 42 | 0 | 0 | 3 | 0 | 0.800618 | 0.224879 | 0.190450 |

## Conclusion

- 운영 정책은 계속 `v2`입니다.
- `v3_candidate_p95`는 false auto가 많아 운영에 부적합합니다.
- `v3_geom_p95`는 false auto를 제거했지만, interpolated auto를 다시 열 근거는 만들지 못했습니다.