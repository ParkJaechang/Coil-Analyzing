# Release Package Summary

## 이 문서가 무엇을 설명하는지
- 이 문서는 release candidate 기준으로 어떤 실행 파일, 어떤 문서, 어떤 shortcut이 준비됐는지 빠르게 확인하는 용도입니다.
- 일반 실행, 실사용 모드, 대표 예제 preset, validation/retune 문서 패키지를 한 곳에서 묶어 보여줍니다.
- 외부 작업이 남은 항목은 마지막 섹션에서 분리합니다.

## 실행 파일
- app: `D:\programs\Codex\Coil Analyzing\app_field_analysis_quick.py`
- launcher_general: `D:\programs\Codex\Coil Analyzing\launch_quick_lut.cmd`
- launcher_operational: `D:\programs\Codex\Coil Analyzing\launch_quick_lut_operational.cmd`
- launcher_example_bz: `D:\programs\Codex\Coil Analyzing\launch_quick_lut_example_bz_exact.cmd`
- launcher_example_current: `D:\programs\Codex\Coil Analyzing\launch_quick_lut_example_current_exact.cmd`
- launcher_example_finite: `D:\programs\Codex\Coil Analyzing\launch_quick_lut_example_finite_triangle_exact.cmd`

## 문서 패키지
- `quick_start_summary.md`
- `release_candidate_summary.md`
- `operator_workflow.md`
- `operational_scope.md`
- `terminology.md`
- `validation_retune_overview.md`
- `exact_matrix_final.md`
- `lut_catalog.md`
- `lut_audit_report.md`
- `known_limitations.md`
- `next_steps.md`

## 실 validation 상태
- finite exact real validation 1건 완료
- continuous current exact real validation 1건 완료
- continuous field exact real validation 1건 완료
- validation catalog / corrected LUT catalog 반영 완료

## 바탕화면 shortcut
- 일반 실행
- 실사용 모드 실행
- 예제 1 연속 Bz exact
- 예제 2 연속 current exact
- 예제 3 finite triangle exact

## 외부 작업
- bench sign-off
- missing exact 1칸 실측
- interpolated auto 재개
- finite generalization 운영 승격
- > 5 Hz 공식 운영
