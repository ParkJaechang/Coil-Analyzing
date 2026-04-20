# Phase 1 Closeout + Phase 2 Parallel Status

- generated_at_utc: `2026-04-14T05:18:10.584935+00:00`

## 1. bench smoke test 결과

- status: `pending_external_hardware`
- manifest_cases: `3`
- note: `현재 코딩 환경에서는 실제 장비를 제어할 수 없어 manifest/checklist/results template까지만 준비된 상태입니다.`

## 2. exact field 공식 지원 승격 여부

- status: `software_ready_bench_pending`
- remaining_gate: `bench smoke test sign-off`
- export_validation_pass: `True`
- ui_validation_pass: `True`

## 3. finite exact 47 recipes 중 bench 확인 완료한 조합

- exact_recipe_count: `47`
- bench_confirmed_count: `0`
- bench_confirmed_combinations: `[]`
- missing_exact_combinations: `[{'freq_hz': 1.0, 'cycles': 1.0, 'level_pp_a': 20}]`

## 4. steady-state interpolation 개선 전/후 L1FO 비교

### baseline_v2
- policy_version: `v2_continuous_corpus_l1fo`
- case_count: `42`
- auto_count: `0`
- false_auto_count: `0`
- false_block_count: `3`
- mean_shape_corr: `0.800618`
- mean_nrmse: `0.224879`

### localbracket_attempt
- policy_version: `v2_modelside_localbracket_continuous_corpus_l1fo`
- case_count: `42`
- auto_count: `0`
- false_auto_count: `0`
- false_block_count: `3`
- mean_shape_corr: `0.800614`
- mean_nrmse: `0.225882`

### phaseanchor_attempt
- policy_version: `v2_modelside_phaseanchor_continuous_corpus_l1fo`
- case_count: `42`
- auto_count: `0`
- false_auto_count: `0`
- false_block_count: `3`
- mean_shape_corr: `0.800618`
- mean_nrmse: `0.225818`

- conclusion: `이번 model-side interpolation 2회 시도는 baseline 대비 유의미한 quality 개선을 만들지 못했습니다.`

## 5. sine finite generalization 2차 첫 결과

- case_count: `47`
- preview_case_count: `47`
- mean_shape_corr: `0.157136`
- mean_nrmse: `0.384775`
- mean_phase_lag_s: `-0.180646`
- operational_decision: `preview-only 유지`

## 6. 추가 데이터가 있으면 가장 빨라지는 조합 제안

- 1. 1.0 Hz + 1.0 cycle + 20 pp exact recipe 추가 측정
  reason: `현재 exact table의 유일한 결손 조합이며, exact-supported 운영 범위를 바로 메울 수 있음`
- 2. 2.0 Hz / 5.0 Hz 구간에서 sine finite transient 추가 측정
  reason: `stage-2 preview 품질이 가장 나쁜 구간으로 NRMSE와 shape corr가 지속적으로 악화됨`
- 3. 1.0 Hz / 1.25 Hz 구간에서 0.75, 1.5 cycle 추가 반복 측정
  reason: `중간 대역에서도 preview 일반화 품질이 약하며 phase lag 변동성이 큼`
- 4. 0.25 Hz 대역은 현 exact recipe 중심 유지
  reason: `preview quality가 상대적으로 높아 추가 일반화 ROI가 낮음`

## Export / UI 검증 상태

- ui_cases_all_passed: `True`
- browser_download_validation_passed: `True`
- exact_export_cases: `['continuous_exact_current', 'continuous_exact_field', 'finite_exact_sine']`