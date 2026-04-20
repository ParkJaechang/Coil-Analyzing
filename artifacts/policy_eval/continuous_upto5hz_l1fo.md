# Continuous <= 5 Hz L1FO Re-Evaluation

- generated_at_utc: `2026-04-14T05:50:52.558176+00:00`
- official_support_band_hz: `0.25 ~ 5`

## Official Continuous Scope

- usable_auto: `continuous + current target + exact support only, <= 5 Hz`
- preview_only: `['continuous + current target + interpolated_in_hull', 'continuous + field target + interpolated_in_hull', 'continuous exact support above 5 Hz']`

## Model-Side Compare (Pre vs Post <= 5 Hz Refit)

| dataset | case_count | auto_count | false_auto | false_block | mean_shape_corr | mean_nrmse | mean_phase_lag_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all-band subset reference | 38 | 0 | 0 | 3 | 0.787036842532 | 0.23225973485534224 | -0.2618215460526316 |
| <=5 Hz refit | 30 | 0 | 0 | 3 | 0.8335029387398395 | 0.20042313166364503 | -0.14231770833333332 |

- delta(mean_shape_corr): `0.04646609620783948`
- delta(mean_nrmse): `-0.031836603191697205`
- delta(mean_phase_lag_cycles): `0.11950383771929826`

## Bench Case Proposal (<= 5 Hz)

- continuous exact current: freq=0.5 Hz (inside official band and already matched by the validated exact-current export bundle)
- continuous exact field: freq=0.25 Hz (inside official band and already matched by the validated exact-field export bundle)
- finite exact sine: freq=0.5 Hz, cycles=1, level=20 pp (inside official band and already matched by the validated finite exact export bundle)

- conclusion: `Narrowing the steady-state evaluation corpus to <= 5 Hz improves product focus, but interpolated preview quality still needs model-side improvement before any auto rollout can be reconsidered.`