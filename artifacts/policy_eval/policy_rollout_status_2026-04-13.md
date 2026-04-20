# Policy Rollout Status

- generated_at_utc: `2026-04-13T11:54:25.184250+00:00`
- evaluation_mode: `leave-one-frequency-out`
- promotion_scope_under_review: `interpolated_in_hull + current target`
- field_target: `hold`

## Summary

| scope | case_count | auto_count | false_auto | false_block | correct_auto | correct_block | margin_source | support_state_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full/v2 | 42 | 0 | 0 | 3 | 0 | 39 | gain | {'interpolated_in_hull': 42} |
| full/v3_candidate_p95 | 42 | 18 | 17 | 2 | 1 | 22 | p95 | {'interpolated_in_hull': 42} |
| sine/v2 | 21 | 0 | 0 | 1 | 0 | 20 | gain | {'interpolated_in_hull': 21} |
| sine/v3_candidate_p95 | 21 | 18 | 17 | 0 | 1 | 3 | p95 | {'interpolated_in_hull': 21} |
| triangle/v2 | 21 | 0 | 0 | 2 | 0 | 19 | gain | {'interpolated_in_hull': 21} |
| triangle/v3_candidate_p95 | 21 | 0 | 0 | 2 | 0 | 19 | p95 | {'interpolated_in_hull': 21} |

## Decision

- Operational policy remains `v2`.
- `v3_candidate_p95` is not eligible for promotion on the larger continuous corpus because it introduces false auto cases.
- Full corpus result: `v2 false_auto=0`, `v3 false_auto=17`.
- `v3_candidate_p95` auto count rises to `18`, but only `1` of those are correct auto cases.
- Sine is the failure driver: `false_auto=17` on `21` interpolated sine cases.
- Triangle remains preview-only in practice: `v3 auto_count=0`, `false_auto=0`.

## Margin Audit

- `v2` policy margin source: `gain`; mean gain/peak/p95 margins = `0.00%` / `73.94%` / `74.08%`.
- `v3_candidate_p95` margin source: `p95`; mean gain/peak/p95 margins = `0.00%` / `73.94%` / `74.08%`.
- The p95 margin is materially less conservative than the legacy gain margin, but it is still not sufficient as a promotion rule on its own.

## UI Validation

- method: `streamlit.testing.v1.AppTest`
- limitation: `Programmatic UI validation only; no real browser automation in current environment.`
- limitation: `Current AppTest API does not expose download_button elements directly.`

| name | support_state | auto_state | engine | warning |
| --- | --- | --- | --- | --- |
| exact_current_auto | exact | 가능 | harmonic_surface | 이 validation run은 자동 기본 후보 조건을 만족하지 않습니다. 사유: frequency too far |
| interpolated_current_preview | interpolated_in_hull | preview-only | harmonic_surface | preview only: exact frequency support 없음 / predicted_error_band_above_threshold / input_limit_margin_below_threshold / surface_confidence_below_threshold | 이 validation run은 자동 기본 후보 조건을 만족하지 않습니다. 사유: frequency too far |
| interpolated_current_blocked |  |  |  | 선택한 파형/주파수 조합으로 파형 보정용 모델을 만들 수 없습니다. |

## Operational Scope

- Promote now: none beyond current `exact` auto path.
- Keep as preview-only: `interpolated_in_hull + current target`.
- Keep blocked/held: `field target` auto promotion, `interpolated_edge`, `out_of_hull`.
