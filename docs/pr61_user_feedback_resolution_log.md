# PR61 사용자 피드백 / 수정 이력 / 개발 주의사항 로그

이 문서는 PR61 `codex/finite-feedback-cycle-policy-backend`에서 사용자가 런타임으로 확인하며 제기한 문제, 수정된 내용, 아직 주의해야 할 정책을 누적 기록하는 living document입니다.

목적:
- 사용자 런타임 피드백을 테스트 통과 여부와 별도로 추적한다.
- 같은 문제가 반복되지 않도록 원인과 금지 패턴을 명확히 남긴다.
- 다음 hotfix 때 “무엇을 고쳐야 하는지”보다 “왜 그렇게 해야 하는지”를 먼저 확인하게 한다.
- 코드/테스트/PR body가 통과해도 화면에서 보이지 않으면 완료가 아니라는 기준을 유지한다.

업데이트 규칙:
- 사용자가 새 문제를 지적하면 `신규 피드백 로그`에 추가한다.
- 수정이 끝나면 `수정 상태`를 `fixed`, `partial`, `blocked`, `deferred` 중 하나로 갱신한다.
- 재발 가능성이 있는 항목은 `개발 주의사항`에 정책으로 승격한다.
- runtime evidence가 없으면 `verified_runtime = no`로 남긴다.
- 테스트만 통과한 상태는 `verified_tests = yes`, `verified_runtime = no`로 분리한다.
- generated artifact, local upload cache, export cache는 이 문서에 링크하거나 요약만 남기고 커밋하지 않는다.

## 현재 핵심 정책

모델링 공통:
- 목표 자기장 개형은 `fixed_rounded_triangle`이다.
- 목표 피크 자기장은 사용자 설정값이다.
- 내부 field normalization은 목표 피크 자기장 기준으로 수행한다.
- HallBz convention은 `effective field = -HallBz raw`이다.
- 전압 command limit / normalization 기준은 현재 `±10V`이다.
- Fourier/harmonic resynthesis는 사용하지 않는다.
- heavy calculation은 버튼 기반으로만 실행한다.
- 최종 LUT export 컬럼은 `sample_index,time_s,voltage_v`만 허용한다.

Finite mode:
- production cycle은 `1.0` / `1.5` cycle이다.
- finite 1.0 cycle phase sync 기준은 첫 번째 양의 피크이다.
- finite 1.5 cycle phase sync 기준은 음의 피크이다.
- startup transient를 그대로 command에 보존하는 legacy 방식은 기본이 아니다.
- tail / zero-return은 finite 전용 옵션이다.
- tail OFF는 active 이후 tail만 끄는 것이며 active correction을 끄면 안 된다.

Continuous mode:
- production은 steady-state `1cycle` 반복 출력용이다.
- 1.5cycle production 금지.
- zero-return tail OFF.
- startup transient 제외.
- 마지막 stop 영향 cycle은 기본 선택하지 않는다.
- source waveform family는 모델링 입력 데이터 선택 기준이고, target field shape와 혼동하면 안 된다.

데이터 / upload:
- physical storage filename과 canonical/original filename을 분리한다.
- hash prefix가 붙은 저장 파일명은 내부 충돌 방지용이다.
- parser, UI label, waveform/frequency/cycle inference는 canonical filename 기준이어야 한다.
- 파일 삭제/rename은 사용자 승인 전 금지.

## 신규 피드백 로그

### 2026-05-27: 전압 제한 5V를 10V로 상향

사용자 피드백:
- “5V limit들 전부 10V로 상향시켜줘.”

원인:
- core modeling default, review normalization, UI label, metadata, 테스트 계약에 `±5V`, `peak_to_5V`, `peak_to_5V_or_limit`가 넓게 남아 있었다.
- 일부 경로는 실제 계산은 바뀌었지만 full CI 테스트에는 `<=5.0` 기대값이 남아 있었다.

수정:
- `src/field_analysis/voltage_policy.py` 추가.
- `COMMAND_VOLTAGE_LIMIT_V = 10.0`
- `COMMAND_VOLTAGE_LIMIT_LABEL = "±10V"`
- `COMMAND_VOLTAGE_NORMALIZATION_MODE = "peak_to_10V"`
- `COMMAND_VOLTAGE_NORMALIZATION_OR_LIMIT_MODE = "peak_to_10V_or_limit"`
- finite / continuous 1차, finite 2차, feedback correction, tail, raw waveform review, final LUT review UI를 공통 정책으로 연결.
- `src/tests`에서 `±5V`, `peak_to_5V` 잔여 참조 제거.

검증:
- `python -m pytest -q`: `498 passed, 253 warnings`
- GitHub CI: `success x2`

수정 상태:
- fixed

개발 주의사항:
- 전압 정책은 하드코딩하지 말고 `voltage_policy.py`를 사용한다.
- `5.0` 숫자는 frequency/step/기타 의미일 수 있으므로 grep 결과를 맥락별로 구분한다.
- UI label, metadata, export review, raw waveform label까지 같이 바꿔야 한다.

## 누적 이슈 / 피드백 / 처리 요약

### 1. Continuous source frequency mismatch와 stale selection

사용자 피드백:
- Continuous source dataset은 잡히지만 steady-state 1cycle 추출 시 frequency mismatch 또는 generic invalid message가 반복됐다.
- 어떤 source file이 선택됐는지, source frequency가 무엇인지, Quick LUT target frequency가 무엇인지 알 수 없었다.
- UI target frequency를 바꿨는데 selectbox가 이전 file을 유지하는 것으로 의심됐다.

원인:
- source frequency inference가 preamble/attrs 중심이고 filename inference가 부족했다.
- target frequency 기준 candidate filtering / auto-selection / mismatch display가 부족했다.
- selectbox state가 target 변경 후 stale selection을 유지할 수 있었다.

수정 방향:
- source frequency inference priority:
  - preamble `Frequency(Hz)`
  - filename pattern
  - frame attrs / metadata
  - normalized frame column `freq_hz`
  - user attribution
- candidate list에 file, source category, source_freq_hz, frequency source, schema status 표시.
- exact target frequency match를 우선 정렬 / auto-select.
- target freq 변경 시 selection signature를 바꾸고 stale selection reset.
- mismatch는 production block하되 source/target/error%를 표시.

수정 상태:
- partial/fixed mixed

재발 방지:
- source metadata가 target config를 절대 overwrite하면 안 된다.
- mismatch는 target 변경이 아니라 warning/block이다.
- runtime debug panel에 target/source/selected candidate/match status를 표시한다.

### 2. Continuous source waveform family 선택 누락

사용자 피드백:
- sine/triangle continuous 데이터가 둘 다 있는데 선택 옵션 없이 triangle이 자동 선택되는 것처럼 보였다.
- finite modeling은 주로 sine LUT를 사용했으므로 source waveform family 선택이 명확해야 했다.
- 이후 정책 변경으로 source/input waveform family 기본값은 triangle로 정리됐다.

원인:
- source waveform family inference와 filter UI가 없거나 약했다.
- target field shape와 source voltage waveform family가 UI에서 혼동됐다.

수정 방향:
- source/input waveform family selector 추가.
- 옵션: `triangle`, `sine`, `rounded_triangle`, `auto`, `all/review`.
- 현재 정책 기본값: `triangle`.
- target field shape는 항상 `fixed_rounded_triangle`.
- candidate label에 canonical filename, waveform family, freq, cycle, category, schema status 표시.

수정 상태:
- partial/fixed mixed

재발 방지:
- “목표 자기장 개형”과 “source/input waveform family”를 같은 개념으로 표시하지 않는다.
- auto-select는 target freq + source family match 기준이어야 한다.

### 3. Continuous steady-state 1cycle extraction과 1차 modeling 연결 부족

사용자 피드백:
- Steady-state 1cycle preview만 있고 실제 1차 modeling이 이어지지 않는 것처럼 보였다.
- Continuous 1차 modeling 버튼은 valid extraction이 있으면 사용하고, 없거나 dirty면 extraction부터 실행해야 했다.

원인:
- extraction result와 first modeling result의 session_state/result contract가 분리되어 있었다.
- command_profile 생성 실패와 preview 성공이 UI에서 혼동됐다.

수정 방향:
- `continuous_steady_state_extraction_result`
- `quick_lut_first_model_result_continuous`
- `quick_lut_first_model_result_continuous_metadata`
- command_profile non-empty, limited_voltage_v 존재, loop-safe 1cycle 조건을 success로 사용.
- invalid extraction result는 success 상태로 저장하지 않음.

수정 상태:
- partial/fixed mixed

재발 방지:
- plot/preview와 modeling success를 분리한다.
- command_profile이 없으면 success message를 표시하지 않는다.

### 4. Continuous terminal cycle / stop 영향

사용자 피드백:
- selected cycle이 거의 마지막 steady cycle이라 output stop/decay 영향이 field 후반에 들어왔다.
- phase delay 때문에 마지막 cycle을 쓰면 field support가 stop 이후 decay를 참조한다.

원인:
- representative cycle selection이 last stable cycle 중심이었다.
- command_stop_s / terminal guard / field_support_end_s 조건이 부족했다.

수정 방향:
- representative mode: `last_stable_non_terminal_cycle`
- `terminal_guard_cycle_count = 1`
- field_support_end_s <= command_stop_s 조건을 만족하는 이전 stable cycle 선택.
- command stop detection 추가.

수정 상태:
- partial/fixed mixed

재발 방지:
- “stable metric이 좋다”와 “modeling support가 안전하다”는 별개다.
- phase alignment용 support window가 stop/decay 영역을 참조하면 안 된다.

### 5. Continuous phase alignment support crop

사용자 피드백:
- measured field smoothed/aligned trace가 output 1cycle 끝까지 이어지지 않았다.
- residual/correction 후반이 계산되지 않았다.
- output 1cycle crop 안에서만 phase shift하면 후반 source가 부족해지는 것이 원인이라고 지적했다.

원인:
- output command grid와 measured support grid가 분리되지 않은 경로가 있었다.
- selected 1cycle crop 후 smoothing/alignment를 수행하면 phase shift 후 뒤쪽 source가 부족해졌다.

수정 방향:
- native source support 전체에서 smoothing.
- output grid는 1cycle endpoint-exclusive.
- aligned measured:
  - `aligned_source_time = output_time + selected_voltage_start_s + phase_delay_s`
  - native smoothed measured에서 interpolation.
- support 부족 시 block/warning, NaN-to-zero 금지.

수정 상태:
- partial/fixed mixed

재발 방지:
- phase shift를 적용하는 모든 경로는 output grid와 source support grid를 분리해야 한다.
- active/output 구간 내부 NaN residual을 0 correction으로 바꾸면 안 된다.

### 6. Continuous final LUT export 누락 / duplicate key

사용자 피드백:
- Continuous 1차 modeling command는 생성되지만 finite처럼 최종 전압 LUT export UI가 없었다.
- Continuous 1차/2차 결과를 선택해서 `sample_index,time_s,voltage_v` 형식으로 다운로드해야 했다.
- 이후 `StreamlitDuplicateElementKey`가 발생했다.

원인:
- continuous result contract와 final export source list가 finite와 달랐다.
- 동일 export section이 여러 위치에서 렌더되며 같은 widget key를 사용했다.

수정 방향:
- Continuous 최종 전압 LUT 추출 섹션 추가.
- 1차/2차 result option 표시.
- 2차 없으면 unavailable, 1차 export는 가능.
- key namespace 인자 추가로 duplicate key 방지.

수정 상태:
- fixed

재발 방지:
- Streamlit reusable component에는 반드시 `key_namespace`를 제공한다.
- 같은 UI section이 두 번 렌더될 수 있는 경로를 고려한다.

### 7. Continuous export endpoint not period exclusive

사용자 피드백:
- continuous final export에서 `continuous_endpoint_not_period_exclusive`가 발생했다.

원인:
- loop-safe 1cycle export 검증이 period-exclusive 조건을 엄격히 요구했다.
- command profile time_s max가 period_s와 같거나 period estimate와 mismatch일 수 있었다.

수정 방향:
- continuous output은 `0 <= time_s < period_s`.
- endpoint duplicate 금지.
- export helper에서 loop-safe validation을 명확히 하고 reason 표시.

수정 상태:
- partial/fixed mixed

재발 방지:
- continuous export는 finite export와 달리 loop boundary 중복 샘플을 허용하지 않는다.

### 8. Finite tail policy / 2Hz 이상 auto OFF

사용자 피드백:
- finite 2차 command plot에서 tail 구간 자기장이 급격히 내려갔다 올라왔다.
- 2Hz 이상에서는 phase delay 때문에 마지막 피크 형성 전에 역전압이 걸릴 수 있어 tail이 해롭다.
- tail을 주파수별로 자동 OFF하고 수동 override가 필요했다.

원인:
- post-cycle zero-return tail이 주파수에 무관하게 적용될 수 있었다.
- tail OFF가 active correction까지 자르는 회귀가 있었다.

수정 방향:
- `resolve_finite_tail_policy(freq_hz, mode, threshold_hz)`
- auto policy:
  - `freq_hz < threshold`: enabled
  - `freq_hz >= threshold`: disabled
- default threshold 2.0Hz, UI 조정 가능.
- tail OFF:
  - active-only output
  - tail mask false
  - tail trace hidden
  - export active-only
- threshold 문구는 고정 “2Hz 이상” 대신 현재 threshold 기반 동적 문구.

수정 상태:
- fixed/partial mixed

재발 방지:
- tail OFF는 active correction OFF가 아니다.
- active residual/correction support는 tail sample 생성 여부와 독립이다.

### 9. Finite active-end residual/correction 끊김

사용자 피드백:
- finite 2차/1차 phase-aligned residual이 active 끝까지 계산되지 않아 command 끝부분이 꺾였다.
- tail OFF 후 active-only output으로 줄면서 post-active measurement support가 빠진 것으로 보였다.

원인:
- output command grid와 measurement support grid가 혼동됐다.
- active 내부 non-finite residual이 0으로 채워질 수 있었다.

수정 방향:
- native actual-drive/source timebase에서 smoothing.
- active output time grid에 대해:
  - `measured_for_second(t) = native_smoothed_measured(t + phase_shift_s)`
- required source end:
  - `active_end + max(phase_shift_s, 0) + margin`
- active 내부 NaN residual zero-fill 금지.
- active_end_kink diagnostic 추가.

수정 상태:
- fixed/partial mixed

재발 방지:
- tail OFF와 active correction support를 연결하지 않는다.
- active end missing residual은 repair/zero-fill하지 않는다.

### 10. Finite 1차 phase sync 도입

사용자 피드백:
- Continuous 1차와 finite 2차처럼 finite 1차도 phase/peak sync 기반으로 바꿔야 했다.
- 기존 startup delay-preserving 방식은 파형 왜곡이 남았다.

원인:
- finite 1차가 startup delay를 포함한 measured response를 target과 비교하는 방식이었다.

수정 방향:
- finite 1차 기본 mode: `phase_synced`
- legacy delay-preserving은 review-only.
- measured smoothing → phase sync → residual → correction_delta → command.
- source/input waveform default는 triangle.

수정 상태:
- partial/fixed mixed

재발 방지:
- legacy trace와 phase_synced trace를 섞지 않는다.
- finite first result session_state는 active mode를 명확히 표시한다.

### 11. Finite 1차 measured source가 target/reference처럼 보이는 문제

사용자 피드백:
- finite 1차 phase sync plot에서 target과 measured가 너무 완벽히 겹쳐 실제 측정 field가 아닌 것 같았다.
- support reference, target, old modeled result, continuous data가 섞였을 가능성이 있었다.

원인:
- measured source identity validation이 부족했다.
- source frame origin을 kernel에 명확히 전달하지 않았다.

수정 방향:
- finite first measured source는 실제 finite LUT field column만 허용.
- target/support_reference/predicted/previous modeled/continuous source 금지.
- source identity summary 표시:
  - source file
  - measured field column
  - source data origin
  - actual measured yes/no
- suspicious measured==target diagnostic 추가.

수정 상태:
- partial/fixed mixed

재발 방지:
- model kernel input contract에 `source_data_origin`을 요구한다.
- reference trace는 measured trace로 쓰지 않는다.

### 12. Support reference를 target end에서 끊는 문제

사용자 피드백:
- support reference를 target end에서 끊으니 phase sync 후 계속 뒤에서 끊긴다.
- 실측 데이터에는 target end 뒤 데이터가 있으므로 전체를 받아야 한다.
- support reference는 0으로 수렴할 때까지 plot에 넣어야 한다.

원인:
- target-aligned support/reference를 plot/modeling input으로 쓰는 경로가 있었다.
- native support source를 target window 기준으로 잘라 쓰는 회귀가 있었다.

수정 방향:
- 원본 실측 support 전체를 DataFrame으로 유지.
- target time으로 자른 reference를 phase sync 입력으로 사용하지 않음.
- nonzero start/end detection으로 앞 idle 구간 제거.
- target_end 뒤 source support도 plot/debug에 표시.

수정 상태:
- partial/fixed mixed

재발 방지:
- support reference는 “검토/plot용”과 “modeling input source”를 분리한다.
- phase sync 입력은 native measured support source를 우선 사용한다.

### 13. Support reference 앞 idle 구간 포함 문제

사용자 피드백:
- 제대로 된 support reference를 가져왔지만 앞 쉬는 구간이 그대로 들어갔다.
- 파형 시작 지점을 감지하고 앞 idle 구간을 날려 앞으로 당겨야 한다.

원인:
- raw support timebase를 그대로 표시/사용하면서 nonzero start alignment가 부족했다.

수정 방향:
- support/reference source의 voltage 또는 field nonzero start 감지.
- `support_reference_source_window_start_s` 기준으로 output/native time mapping.
- 앞 idle 구간은 plot/modeling alignment에서 제외.

수정 상태:
- partial/fixed mixed

재발 방지:
- raw file absolute time과 modeling local time을 구분한다.
- plot x축은 필요 시 local time으로 재정렬한다.

### 14. Phase sync peak 기준

사용자 피드백:
- 처음에는 dominant peak와 가장 가까운 같은 극성 voltage peak 기준이 제안됐지만, finite 정책은 명확히 해야 했다.
- finite 1.5cycle은 음의 피크 기준.
- finite 1.0cycle은 첫 번째 양의 피크 기준으로 변경.

원인:
- phase sync 기준이 “첫 피크”, “dominant peak”, “같은 극성 nearest peak” 사이에서 흔들렸다.

현재 정책:
- finite 1.0cycle: first positive peak sync.
- finite 1.5cycle: negative peak sync.
- continuous: voltage peak ↔ measured field peak 기준의 continuous loop-safe 정책 유지.

수정 상태:
- fixed/partial mixed

재발 방지:
- phase sync 기준은 cycle_count에 따라 명시한다.
- metadata에 peak polarity / selected peak index / peak time을 남긴다.

### 15. Field normalization 의미 혼동

사용자 피드백:
- “±50mT 안으로 들어오라”가 아니라 어느 쪽이든 최대 피크가 target peak가 되도록 scale-only normalization하라는 의미였다.
- offset을 임의로 목표 피크로 당기는 방식은 안 된다.
- 실측값을 그대로 쓰고 증폭기 gain은 고정되어 있으므로 gain 기반 임의 보정은 제거/축소해야 했다.

원인:
- baseline removal, offset center, peak-to-50mT scaling, target peak scaling 의미가 UI/코드에서 혼동됐다.
- residual 반영 gain이 과도해 보였다.

수정 방향:
- measured field는 raw/effective를 보존.
- scale-only normalization:
  - dominant abs peak 기준으로 target peak에 맞춤.
- target field도 user target peak에 맞게 생성되어야 함.
- residual은 target_scaled - measured_aligned_scaled.
- correction delta는 기존 계산 로직 기반으로 안정화.

수정 상태:
- partial/fixed mixed

재발 방지:
- offset shift와 amplitude scale을 혼동하지 않는다.
- target peak를 바꾸면 target field, measured normalization, base voltage scaling, correction delta가 모두 일관되게 바뀌어야 한다.

### 16. Target peak 변경 시 target field가 안 바뀌는 문제

사용자 피드백:
- target peak를 바꿨는데 target field가 안 변하고 measured만 scale되면 residual이 틀어진다.
- correction_delta_v도 target peak에 따라 바뀌어야 한다.

원인:
- 일부 경로에서 target template이 내부 고정 50mT 또는 fixed 100pp 의미로 남아 있었다.
- target peak field와 internal normalization reference가 분리되지 않았다.

수정 방향:
- user target peak field를 target config source-of-truth에 저장.
- target template 생성 시 target peak 반영.
- measured scale, residual, correction metadata에 target_peak_field_mT 포함.

수정 상태:
- partial/fixed mixed

재발 방지:
- target peak 변경 테스트는 target trace, residual, correction_delta, command voltage 모두 확인해야 한다.

### 17. 원본 입력 전압과 모델링 입력 전압 구분

사용자 피드백:
- 1차 command plot에 기존 입력 전압도 같이 띄워 비교하고 싶다.
- 기존 입력 전압은 이상적인 삼각파가 아니라 실제 1차 구동 LUT의 입력 전압이어야 한다.
- 원본 ±5V 입력 전압을 field normalization scale만큼 줄인 modeling input voltage도 표시해야 한다.

원인:
- source/base voltage, ideal/reference voltage, corrected command가 plot에서 혼동됐다.
- 일부 plot은 이상적인 삼각파 또는 잘못된 reference를 “기존 입력 전압”처럼 표시했다.

수정 방향:
- command plot trace:
  - 원본 입력 전압
  - 모델링 입력 전압
  - correction_delta_v
  - 최종 command
- finite도 continuous와 같은 형식으로 표시.

수정 상태:
- partial/fixed mixed

재발 방지:
- “원본 입력 전압”은 raw waveform / first drive LUT에서 온 voltage column이어야 한다.
- ideal template voltage를 원본 입력 전압으로 표시하지 않는다.

### 18. Continuous voltage normalization이 입력 전압 그대로 보정되는 문제

사용자 피드백:
- continuous는 입력전압을 그대로 보정하고 있었다.
- field가 정규화된 만큼 원본 입력 전압도 정규화/scale되어야 출력 자기장 scale 문제가 생기지 않는다.
- 목표 피크를 바꾸면 continuous도 해당 피크 기준으로 모델링해야 한다.

원인:
- continuous first modeling에서 base/source voltage normalization target이 고정값이거나 raw voltage 그대로 사용되는 경로가 있었다.
- target peak field와 voltage scaling이 충분히 연결되지 않았다.

수정 방향:
- continuous도 finite와 같은 logic:
  - measured field scale
  - source voltage scale
  - base/modeling input voltage
  - residual
  - correction_delta
  - final command
- continuous extraction만 다르고 modeling normalization 의미는 finite와 맞춘다.

수정 상태:
- partial/fixed mixed

재발 방지:
- continuous/finite normalization helper를 가능한 공통화한다.
- continuous plot에서 source/base/correction/final을 모두 표시한다.

### 19. Target config가 1Hz/1cycle을 1.5로 바꾸는 것처럼 보이는 문제

사용자 피드백:
- 사용자가 1Hz 조건으로 1차 모델링을 실행하고 싶은데 앱이 1.5로 바꾸는 것처럼 보였다.
- frequency 1Hz가 1.5Hz로 바뀌는지, cycle 1.0이 1.5cycle로 바뀌는지 UI상 혼동됐다.

원인:
- target frequency와 target cycle count source-of-truth가 명확하지 않았다.
- source dataset metadata/cached result가 UI target을 덮어쓸 위험이 있었다.

수정 방향:
- `quick_lut_target_config`
- `quick_lut_applied_target_config`
- UI current config와 applied config 분리.
- source metadata mismatch는 warning/block, target overwrite 금지.
- finite cycle selection 유지.
- continuous mode는 cycle만 1.0으로 lock, frequency는 보존.

수정 상태:
- partial/fixed mixed

재발 방지:
- target_freq_hz와 target_cycle_count를 UI에서 따로 표시한다.
- result metadata에는 modeled_target_config_snapshot을 저장한다.

### 20. Upload memory / cache restore 문제

사용자 피드백:
- 이전에는 continuous cycle 데이터가 Global upload memory에 잡혔는데 갑자기 `No cached uploads`로 보였다.
- 연속 cycle 입력 요약이 empty가 됐다.

원인:
- upload memory restore 순서, category alias, manifest active entry와 disk file mismatch가 있었다.
- continuous/transient files에는 hash prefix가 있었고 parser/canonical filename 처리가 불충분했다.

수정 방향:
- upload category alias normalization:
  - `continuous`, `continuous-cycle`, `continuous_cycle`, `연속 cycle`, `continuous_steady_state` 등.
- disk cache / manifest restore를 session_state empty 상태에서도 수행.
- storage filename과 canonical filename 분리.
- old hash-prefixed files compatibility layer.

수정 상태:
- fixed/partial mixed

재발 방지:
- upload memory reset은 명시 버튼 외에는 하지 않는다.
- candidate discovery는 restore 이후 실행한다.

### 21. Hash-prefixed filename / canonical filename 문제

사용자 피드백:
- upload 폴더의 continuous/transient 파일명 앞에 긴 hash prefix가 붙어 parser가 실패하는 것으로 보였다.
- 2nd 폴더는 prefix가 없어 저장 방식이 달랐다.

원인:
- physical storage filename을 parser/source identity로 사용했다.

수정 방향:
- metadata:
  - storage_path
  - storage_filename
  - original_filename
  - canonical_filename
  - upload_category
  - upload_id/hash
- canonical filename recovery:
  - `^[0-9a-f]{8,32}_` prefix strip
  - strip 결과가 known finite/continuous pattern일 때만 적용.
- UI label과 parser는 canonical filename 사용.

수정 상태:
- fixed/partial mixed

재발 방지:
- physical filename을 source identity로 쓰지 않는다.
- file rename/delete 금지.

### 22. UI cleanup / legacy wording

사용자 피드백:
- 예전 UI 기능이 다시 살아나거나 섞여 있었다.
- `실구동 결과 CSV 업로드`, `피드백 run`, `Bz_mT waveform compensation`, `목표 출력 외삽`, `필요 AMP gain`, `DAQ 전압`, `Support/Provenance/Consistency` 등이 혼란스러웠다.
- 영어 문구가 많고 정보가 과다했다.

원인:
- legacy diagnostic UI가 main workflow에 노출됐다.
- current modeling semantics와 old calibration semantics가 섞였다.

수정 방향:
- main UI에는:
  - 데이터 source 요약
  - target 설정
  - modeling mode
  - 1차 modeling 실행/결과
  - 2차 correction 실행/결과
  - final LUT export
- legacy upload / hardware calibration / support provenance는 Advanced/Legacy expander로 이동.
- 사용자-facing label 한글화.

수정 상태:
- partial

재발 방지:
- main UI에 debug JSON/raw metadata dump를 노출하지 않는다.
- 영어 internal key는 Debug expander에만 둔다.

### 23. Target semantics: 100mT pp fixed 제거

사용자 피드백:
- `목표 bz_mT PP 100.000 mT`, `100mT pp fixed`, `fixed 100pp` 문구가 남아 있었다.
- 목표 자기장 “개형”만 fixed이고, 목표 피크는 사용자 설정값이어야 했다.

원인:
- old field-only modeling 문구가 UI에 남아 있었다.
- target peak semantics와 internal normalization reference가 혼동됐다.

수정 방향:
- main UI 세 가지 분리:
  - 목표 자기장 개형: fixed rounded triangle
  - 목표 피크 자기장: 사용자 설정값
  - 내부 모델링 정규화 기준: target peak / 또는 명시 기준
- `100mT pp fixed` user-facing text 제거.

수정 상태:
- partial/fixed mixed

재발 방지:
- fixed는 shape에만 붙인다.
- peak/pp를 fixed라고 쓰지 않는다.

### 24. Target rounded triangle ripple

사용자 피드백:
- physical target rounded triangle의 직선 구간이 울룩불룩해 보였다.

원인:
- target generation/resampling/smoothing 과정에서 직선 구간 ripple이 생길 수 있었다.

수정 방향:
- analytic fixed rounded triangle template.
- rounded corner만 smoothing.
- linear segment deviation diagnostic.
- finite/continuous 동일 target template 사용.

수정 상태:
- fixed/partial mixed

재발 방지:
- target template에 smoothing 필터를 전체 적용하지 않는다.
- template quality test와 runtime diagnostic을 함께 유지한다.

### 25. 2차 보정 입력 source UI 폐기

사용자 피드백:
- “2차 보정 입력 source” UI 부분은 폐기됐으니 삭제해야 했다.
- 1차 구동 결과 폴더 기반으로 자동 로드하는 방향.

현재 사용자가 지정한 폴더:
- `D:\programs\Codex\Coil Analyzing_clean\outputs\field_analysis_app_state\uploads`
- `Continuous_1st_Result`
- `Transient_1st_Result`

정책:
- finite actual-drive는 transient 폴더로 잡는다.
- continuous actual-drive는 continuous 1st result 폴더로 잡는다.

수정 상태:
- partial

재발 방지:
- “2nd folder” 용어와 “1차 구동 결과 folder” 용어를 혼동하지 않는다.
- 2차 modeling input은 1차 command가 아니라 1차 실제 구동 measurement이다.

### 26. Finite UI 정보 과다 / 한글화

사용자 피드백:
- finite cycle Quick LUT UI에 정보가 과다하고 영어가 많다.
- 뭔지 모를 문구가 많아 한글화가 필요하다.

수정 방향:
- main view:
  - 핵심 상태 카드
  - source/target/match
  - phase sync summary
  - 1차 command plot
  - export
- Debug:
  - support ids
  - route
  - guardrail metrics
  - raw metadata

수정 상태:
- partial

재발 방지:
- 새 diagnostic을 추가할 때 main 화면에 바로 추가하지 말고 Debug expander부터 시작한다.

## 개발 주의사항

### A. 테스트 통과와 runtime acceptance를 분리한다

주의:
- full pytest pass는 필요조건이지 충분조건이 아니다.
- 사용자가 보는 화면에 반영되지 않으면 완료가 아니다.
- PR body/docs만 업데이트된 상태는 완료가 아니다.

필수 확인:
- UI screenshot/log
- plot trace source
- metadata card
- export CSV preview
- target config summary

### B. source-of-truth를 덮어쓰지 않는다

금지:
- source dataset metadata가 UI target frequency/cycle을 자동 변경.
- cached result metadata가 current target config를 덮어씀.
- upload candidate auto-select가 target_freq_hz를 변경.

허용:
- continuous mode 전환 시 cycle_count만 1.0으로 lock.
- mismatch는 warning/block.

### C. output grid와 support grid를 분리한다

반드시 분리:
- output command grid
- measured/native support grid
- plot/debug support reference grid

금지:
- active/output window로 field를 먼저 자른 뒤 phase shift.
- target end에서 support를 자른 뒤 residual 계산.
- missing support를 0으로 채워 command 생성.

### D. phase sync 기준은 mode/cycle별로 명시한다

현재 정책:
- finite 1.0cycle: first positive peak.
- finite 1.5cycle: negative peak.
- continuous: steady-state 1cycle loop 기준 peak alignment.

metadata 필수:
- phase_sync_method
- selected_peak_polarity
- voltage_peak_time_s
- measured_peak_time_s
- phase_delay_s
- phase_delay_cycles

### E. normalization은 offset 이동이 아니라 scale 정책이다

주의:
- 사용자 의도는 “최대 피크가 목표 피크가 되도록 scale”이다.
- 임의 0기준점/offset shift로 목표에 맞추면 안 된다.
- raw/effective measured field는 보존한다.

확인:
- measured peak before scale
- scale factor
- measured peak after scale
- target peak
- residual uses scaled target and scaled measured consistently

### F. voltage command plot source를 명확히 한다

plot에 표시할 때:
- 원본 입력 전압: raw source/first-drive LUT voltage.
- 모델링 입력 전압: field scale 반영 후 base voltage.
- correction_delta_v: residual 기반 보정량.
- 최종 command: export voltage source.

금지:
- ideal voltage template을 “기존 입력 전압”으로 표시.
- plotted voltage와 export voltage가 다름.

### G. legacy UI는 기본 화면에 두지 않는다

Legacy/Debug로 이동할 것:
- manual actual-drive CSV uploader.
- feedback run labels.
- DAQ/AMP gain/extrapolation.
- Support Reference Provenance.
- Command Prediction Consistency.
- raw JSON metadata dump.

### H. upload/cache는 보존 우선

금지:
- CSV delete.
- physical file rename.
- cleanup/rm/git clean.
- category migration 중 silent drop.

필수:
- canonical filename 표시.
- storage filename은 debug에서만 표시.
- restore status와 category count 표시.

### I. 전압 정책은 `voltage_policy.py`를 사용한다

현재:
- limit: `±10V`
- normalization: `peak_to_10V`
- normalize-or-limit: `peak_to_10V_or_limit`

금지:
- 새 코드에 `5.0`, `10.0` voltage limit 하드코딩.
- UI에 직접 `±10V` 문자열 반복 추가.

### J. result contract를 stage별로 통일한다

모든 exportable result:
- `command_profile: pd.DataFrame`
- `metadata: dict`
- voltage source column 명시
- status 명시

Export CSV:
- `sample_index`
- `time_s`
- `voltage_v`

## 현재 남은 주의/부분 과제

1. Finite UI 한글화와 정보 축소는 계속 진행 필요.
2. `2차 보정 입력 source` 폐기 후 1차 구동 결과 폴더 기반 흐름을 더 명확히 정리해야 함.
3. finite target peak 변경 시 target field, measured scale, correction_delta, command voltage가 모두 같이 바뀌는지 runtime evidence 필요.
4. continuous source discovery UI는 더 단순화/효율화 필요.
5. continuous와 finite normalization 로직의 공통화 여지가 남아 있음.
6. phase sync support grid / output grid 분리 정책은 계속 regression test를 보강해야 함.
7. 기존 docs/reports/cleanup dirty 파일은 이 로그와 별도이며, PR61 hotfix commit 대상이 아님.

## 다음 피드백 업데이트 템플릿

```md
### YYYY-MM-DD: 제목

사용자 피드백:
- 

원인:
- 

수정:
- 

검증:
- tests:
- runtime:
- CI:

수정 상태:
- fixed | partial | blocked | deferred

개발 주의사항:
- 
```
