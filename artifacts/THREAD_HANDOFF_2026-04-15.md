# Thread Handoff - 2026-04-15

## Purpose

This file is the handoff point for a new Codex thread. The next thread should not restart analysis from scratch. It should read the files below first, adopt the current operating decisions as fixed, and continue from the latest verified state.

## Read First

1. [app_field_analysis_quick.py](D:\programs\Codex\Coil Analyzing\app_field_analysis_quick.py)
2. [app_ui.py](D:\programs\Codex\src\field_analysis\app_ui.py)
3. [recommendation_service.py](D:\programs\Codex\src\field_analysis\recommendation_service.py)
4. [compensation.py](D:\programs\Codex\src\field_analysis\compensation.py)
5. [test_recommendation_service.py](D:\programs\Codex\Coil Analyzing\tests\test_recommendation_service.py)
6. [streamlit_case_verification.json](D:\programs\Codex\Coil Analyzing\artifacts\streamlit_case_verification.json)
7. [streamlit_case_verification.md](D:\programs\Codex\Coil Analyzing\artifacts\streamlit_case_verification.md)
8. [streamlit_actual_screen.png](D:\programs\Codex\Coil Analyzing\artifacts\streamlit_actual_screen.png)
9. [streamlit_actual_source.html](D:\programs\Codex\Coil Analyzing\artifacts\streamlit_actual_source.html)

## Current Fixed Operating Scope

- `continuous/current exact <= 5 Hz`: auto operating path only
- `continuous/field exact <= 5 Hz`: software-ready, bench pending
- `finite exact <= 5 Hz`: exact-matrix path only
- `interpolated auto`: disabled
- `preview-only / provisional / unsupported`: keep separated in UI and policy
- `> 5 Hz`: reference-only / not official operating range

## Verified Behavior Already Implemented

- Exact main graph must use `plot_source=exact_prediction`
- Preview must be the only path that shows `Support-Blended Output`
- Recommended command waveform plot uses `limited_voltage_v`, not raw synthesized command
- Finite exact requests should not render support-blended as the main output plot
- Triangle exact requests carry triangle-oriented harmonic weights
- Triangle exact and sine exact command waveforms are distinguishable by FFT

## Important Recent Fixes

### 1. Top route summary card mismatch fixed

The top request summary card in the UI previously said `자동 추천 가능` even when policy gating had already blocked auto download on the same submission. That was fixed in:

- [app_ui.py](D:\programs\Codex\src\field_analysis\app_ui.py)

The recommendation result is now precomputed before route summary rendering, so the top card reflects the actual current result.

### 2. Debug route normalization fixed

`finite provisional` and `finite unsupported` previously leaked an internal legacy `request_route=exact/preview` meaning that did not match the final displayed support state. That was normalized in:

- [recommendation_service.py](D:\programs\Codex\src\field_analysis\recommendation_service.py)

Current intended meaning:

- exact support => `request_route=exact`
- interpolated preview => `request_route=preview`
- provisional finite => `request_route=provisional`
- unsupported => `request_route=unsupported`

### 3. Test expectations updated

- [test_recommendation_service.py](D:\programs\Codex\Coil Analyzing\tests\test_recommendation_service.py)

Current test status at last verified point:

- `59 passed`

## Last Verified UI Cases

These were verified through Streamlit AppTest running the real app UI code path, with stored browser artifacts used as additional render evidence.

### continuous exact current <= 5 Hz

- `request_route=exact`
- `solver_route=harmonic_surface_inverse_exact`
- `plot_source=exact_prediction`
- `support_state=exact`
- no `Support-Blended Output` on the main plot
- auto remains blocked if shape gate fails

### continuous preview interpolated

- `request_route=preview`
- `solver_route=harmonic_surface_inverse_interpolated_preview`
- `plot_source=support_blended_preview`
- `support_state=interpolated_in_hull`
- `Support-Blended Output` shown only here

### finite triangle exact

- exact path should show `plot_source=exact_prediction`
- main plot should use exact predicted output, not support-blended output
- harmonic weights should reflect triangle emphasis

### finite provisional / unsupported

- provisional must remain preview-like in behavior but explicitly labeled as provisional
- unsupported must stay blocked/measurement-needed
- both should not silently masquerade as exact-supported in the UI

## Known Limitation

Live browser submit automation is unreliable in this environment. Do not burn time trying to force Selenium submit if the goal is code-path verification.

Use this order:

1. Streamlit AppTest for per-case execution and assertions
2. Stored browser artifacts for render sanity
3. Selenium only if a very specific browser-only behavior must be captured

## If You Need Fresh Verification

If the next thread changes relevant UI/policy code, it should refresh the case-verification artifacts instead of trusting the old copies blindly.

Minimum refresh targets:

- [streamlit_case_verification.json](D:\programs\Codex\Coil Analyzing\artifacts\streamlit_case_verification.json)
- [streamlit_case_verification.md](D:\programs\Codex\Coil Analyzing\artifacts\streamlit_case_verification.md)

## Copy-Paste Prompt For A New Thread

Use this prompt in the new thread:

```text
이전 쓰레드 작업을 그대로 이어서 진행하세요. 처음부터 다시 분석하지 말고 아래 파일을 먼저 읽고 현재 상태를 기준선으로 채택하세요.

필수로 먼저 읽을 파일:
- D:\\programs\\Codex\\Coil Analyzing\\artifacts\\THREAD_HANDOFF_2026-04-15.md
- D:\\programs\\Codex\\Coil Analyzing\\app_field_analysis_quick.py
- D:\\programs\\Codex\\src\\field_analysis\\app_ui.py
- D:\\programs\\Codex\\src\\field_analysis\\recommendation_service.py
- D:\\programs\\Codex\\src\\field_analysis\\compensation.py
- D:\\programs\\Codex\\Coil Analyzing\\tests\\test_recommendation_service.py
- D:\\programs\\Codex\\Coil Analyzing\\artifacts\\streamlit_case_verification.json
- D:\\programs\\Codex\\Coil Analyzing\\artifacts\\streamlit_actual_screen.png
- D:\\programs\\Codex\\Coil Analyzing\\artifacts\\streamlit_actual_source.html

고정 판단:
- continuous/current exact <= 5 Hz만 auto 운영
- continuous/field exact <= 5 Hz는 software-ready, bench pending
- finite exact <= 5 Hz는 exact-matrix 경로만 사용
- interpolated auto는 계속 닫힘
- preview-only / provisional / unsupported 상태 분리는 유지
- > 5 Hz는 공식 운영 범위가 아님

이미 끝난 수정:
- exact는 support-blended를 메인 그래프에 그리지 않음
- preview에서만 support-blended를 그림
- recommended command waveform plot은 limited_voltage_v를 사용
- 상단 요청 카드가 실제 gating 결과와 일치하도록 수정됨
- finite provisional/unsupported의 debug request_route 의미를 정규화함
- 테스트는 마지막 확인 시 59 passed

우선은 handoff 파일의 마지막 상태를 확인한 뒤, 현재 요청 기준으로 필요한 부분만 이어서 진행하세요.
```

## Recommended Rule For Future Handoffs

When moving to another thread, always point the next thread to this handoff file first. Do not rely on conversational memory alone.
