# Request Router Validation

- generated_at_utc: `2026-04-16T08:45:09.840892+00:00`

| case | freq_before | freq_after | target_after | finite_mode_after | cycle_after | pass |
| --- | --- | --- | --- | --- | --- | --- |
| continuous_preview_apply_exact | 0.75 | 0.5 | 20 | False |  | True |
| finite_provisional_apply_exact | 1 | 1 | 10 | True | 1 | True |
| finite_provisional_apply_provisional | 1 | 1 | 20 | True | 1 | True |
