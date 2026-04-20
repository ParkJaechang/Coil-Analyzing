# Exact And Finite Scope

- 운영 기준은 Bz-first exact-matrix이다.
- continuous/current exact <= 5 Hz는 auto 운영, continuous/field exact <= 5 Hz는 software-ready bench pending이다.
- finite exact <= 5 Hz는 measured exact recipe만 exact로 간주한다.
- sine / 1.0 Hz / 1.0 cycle / 20 pp는 exact가 아니라 provisional preview이며, measured exact 업로드가 들어오면 승격한다.

## Continuous Current Exact Matrix

| waveform | freq_hz | levels_a | cells | status |
| --- | --- | --- | --- | --- |
| sine | 0.25 | 5, 10, 15, 20, 30 | 5 | certified_exact |
| sine | 0.5 | 5, 10, 15, 20, 30 | 5 | certified_exact |
| sine | 1.0 | 5, 10, 15, 20, 30 | 5 | certified_exact |
| sine | 2.0 | 5, 10, 15, 20, 30 | 5 | certified_exact |
| sine | 5.0 | 5, 10, 15, 20 | 4 | certified_exact |
| triangle | 0.25 | 5, 10, 15, 20, 30 | 5 | certified_exact |
| triangle | 0.5 | 5, 10, 15, 20, 30 | 5 | certified_exact |
| triangle | 1.0 | 5, 10, 15, 20, 30 | 5 | certified_exact |
| triangle | 2.0 | 5, 10, 15, 20, 30 | 5 | certified_exact |
| triangle | 5.0 | 5, 10, 15, 20 | 4 | certified_exact |

## Continuous Field Exact Matrix

| waveform | freq_hz | target_levels | status | bench_validation |
| --- | --- | --- | --- | --- |
| sine | 0.25 | variable field target within hardware limits | software_ready_bench_pending | pending |
| sine | 0.5 | variable field target within hardware limits | software_ready_bench_pending | pending |
| sine | 1.0 | variable field target within hardware limits | software_ready_bench_pending | pending |
| sine | 2.0 | variable field target within hardware limits | software_ready_bench_pending | pending |
| sine | 5.0 | variable field target within hardware limits | software_ready_bench_pending | pending |
| triangle | 0.25 | variable field target within hardware limits | software_ready_bench_pending | pending |
| triangle | 0.5 | variable field target within hardware limits | software_ready_bench_pending | pending |
| triangle | 1.0 | variable field target within hardware limits | software_ready_bench_pending | pending |
| triangle | 2.0 | variable field target within hardware limits | software_ready_bench_pending | pending |
| triangle | 5.0 | variable field target within hardware limits | software_ready_bench_pending | pending |

## Continuous Exact Expansion Candidates

| waveform | freq_hz | level_a |
| --- | --- | --- |
| sine | 0.75 | 5.0 |
| sine | 0.75 | 10.0 |
| sine | 0.75 | 20.0 |
| sine | 1.5 | 5.0 |
| sine | 1.5 | 10.0 |
| sine | 1.5 | 20.0 |
| sine | 3.0 | 5.0 |
| sine | 3.0 | 10.0 |
| sine | 3.0 | 20.0 |
| sine | 4.0 | 5.0 |
| sine | 4.0 | 10.0 |
| sine | 4.0 | 20.0 |
| triangle | 0.75 | 5.0 |
| triangle | 0.75 | 10.0 |
| triangle | 0.75 | 20.0 |
| triangle | 1.5 | 5.0 |
| triangle | 1.5 | 10.0 |
| triangle | 1.5 | 20.0 |
| triangle | 3.0 | 5.0 |
| triangle | 3.0 | 10.0 |
| triangle | 3.0 | 20.0 |
| triangle | 4.0 | 5.0 |
| triangle | 4.0 | 10.0 |
| triangle | 4.0 | 20.0 |

## Finite Exact Matrix

| waveform | freq_hz | cycles | levels_pp_a | cells | status |
| --- | --- | --- | --- | --- | --- |
| sine | 0.25 | 0.75 | 10, 20 | 2 | exact |
| sine | 0.25 | 1.0 | 10, 20 | 2 | exact |
| sine | 0.25 | 1.25 | 10, 20 | 2 | exact |
| sine | 0.25 | 1.5 | 10, 20 | 2 | exact |
| sine | 0.5 | 0.75 | 10, 20 | 2 | exact |
| sine | 0.5 | 1.0 | 10, 20 | 2 | exact |
| sine | 0.5 | 1.25 | 10, 20 | 2 | exact |
| sine | 0.5 | 1.5 | 10, 20 | 2 | exact |
| sine | 1.0 | 0.75 | 10, 20 | 2 | exact |
| sine | 1.0 | 1.0 | 10 | 1 | exact |
| sine | 1.0 | 1.25 | 10, 20 | 2 | exact |
| sine | 1.0 | 1.5 | 10, 20 | 2 | exact |
| sine | 1.25 | 0.75 | 10, 20 | 2 | exact |
| sine | 1.25 | 1.0 | 10, 20 | 2 | exact |
| sine | 1.25 | 1.25 | 10, 20 | 2 | exact |
| sine | 1.25 | 1.5 | 10, 20 | 2 | exact |
| sine | 2.0 | 0.75 | 10, 20 | 2 | exact |
| sine | 2.0 | 1.0 | 10, 20 | 2 | exact |
| sine | 2.0 | 1.25 | 10, 20 | 2 | exact |
| sine | 2.0 | 1.5 | 10, 20 | 2 | exact |
| sine | 5.0 | 0.75 | 10, 20 | 2 | exact |
| sine | 5.0 | 1.0 | 10, 20 | 2 | exact |
| sine | 5.0 | 1.25 | 10, 20 | 2 | exact |
| sine | 5.0 | 1.5 | 10, 20 | 2 | exact |
| triangle | 0.25 | 0.75 | 10, 20 | 2 | exact |
| triangle | 0.25 | 1.0 | 10, 20 | 2 | exact |
| triangle | 0.25 | 1.25 | 10, 20 | 2 | exact |
| triangle | 0.25 | 1.5 | 10, 20 | 2 | exact |
| triangle | 0.5 | 0.75 | 10, 20 | 2 | exact |
| triangle | 0.5 | 1.0 | 10, 20 | 2 | exact |
| triangle | 0.5 | 1.25 | 10, 20 | 2 | exact |
| triangle | 0.5 | 1.5 | 10, 20 | 2 | exact |
| triangle | 1.0 | 0.75 | 10, 20 | 2 | exact |
| triangle | 1.0 | 1.0 | 10, 20 | 2 | exact |
| triangle | 1.0 | 1.25 | 10, 20 | 2 | exact |
| triangle | 1.0 | 1.5 | 10, 20 | 2 | exact |
| triangle | 1.25 | 0.75 | 10, 20 | 2 | exact |
| triangle | 1.25 | 1.0 | 10, 20 | 2 | exact |
| triangle | 1.25 | 1.25 | 10, 20 | 2 | exact |
| triangle | 1.25 | 1.5 | 10, 20 | 2 | exact |
| triangle | 2.0 | 0.75 | 10, 20 | 2 | exact |
| triangle | 2.0 | 1.0 | 10, 20 | 2 | exact |
| triangle | 2.0 | 1.25 | 10, 20 | 2 | exact |
| triangle | 2.0 | 1.5 | 10, 20 | 2 | exact |
| triangle | 3.0 | 0.75 | 10, 20 | 2 | exact |
| triangle | 3.0 | 1.0 | 10, 20 | 2 | exact |
| triangle | 3.0 | 1.25 | 10, 20 | 2 | exact |
| triangle | 3.0 | 1.5 | 10, 20 | 2 | exact |

## Missing Exact Cell

| waveform | freq_hz | cycles | level_pp_a | status | current_route | promotion_target |
| --- | --- | --- | --- | --- | --- | --- |
| sine | 1.0 | 1.0 | 20.0 | missing_exact | provisional_preview | 96 exact recipes after measured upload arrives. |

## Provisional Preview Cell

| waveform | freq_hz | cycles | level_pp_a | source_exact_level_pp_a | scale_ratio | status | measured_file_present |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sine | 1.0 | 1.0 | 20.0 | 10.0 | 2.0 | provisional_preview | True |

## Reference-only Continuous Cells (> 5 Hz)

| waveform | freq_hz | levels_a |
| --- | --- | --- |
| sine | 10.0 | 5, 10 |
| sine | 15.0 | 5, 10 |
| triangle | 10.0 | 5, 10 |
| triangle | 15.0 | 5 |

## Intake Robustness Notes

- continuous scan count: `55`
- transient scan count: `96`
- transient waveform aliases accepted: `sine`, `sin`, `sinusoid`, `sinusoidal`, `sinusidal`, `triangle`, `tri`
- continuous ignored files: `0`
- transient ignored files: `0`
