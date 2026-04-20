# Validation Upload Guide

Validation / Retune용 실측 결과 파일은 아래 루트에 넣으면 기억된 업로드처럼 자동 스캔됩니다.

기본 루트:
- `D:\programs\Codex\outputs\field_analysis_app_state\uploads\validation`

권장 하위 폴더:
- `D:\programs\Codex\outputs\field_analysis_app_state\uploads\validation\continuous_current_exact`
- `D:\programs\Codex\outputs\field_analysis_app_state\uploads\validation\continuous_field_exact`
- `D:\programs\Codex\outputs\field_analysis_app_state\uploads\validation\finite_sine_exact`
- `D:\programs\Codex\outputs\field_analysis_app_state\uploads\validation\finite_triangle_exact`

현재 예정된 validation 수집 항목:
- `continuous_current_exact`
  - `continuous_sine_1hz_20pp`
  - `continuous_triangle_1hz_20pp`
- `continuous_field_exact`
  - `continuous_sine_0.5hz_20pp`
  - `continuous_triangle_0.5hz_20pp`
- `finite_sine_exact`
  - `1.25hz_sine_1.25cycle_20pp`
- `finite_triangle_exact`
  - `1.25hz_triangle_1.25cycle_20pp`

동작 규칙:
- 앱은 `validation` 루트 아래 파일을 재귀적으로 스캔합니다.
- 위 하위 폴더 안에 `csv`, `txt`, `xls`, `xlsx`, `xlsm` 파일을 넣으면 다음 실행 또는 persisted upload 로드 시 자동 인식됩니다.
- 파일명/메타데이터가 target 조건과 가까우면 Validation / Retune 탭에서 후보로 자동 연결됩니다.

권장 파일명 예시:
- `continuous_sine_1hz_20pp_validation.csv`
- `continuous_triangle_0.5hz_20pp_field_validation.csv`
- `finite_sine_1.25hz_1.25cycle_20pp_validation.csv`
- `finite_triangle_1.25hz_1.25cycle_20pp_validation.csv`

주의:
- 지원 범위 판정은 파일명만이 아니라 내부 메타데이터와 파싱 결과도 함께 봅니다.
- finite exact validation은 주파수와 레벨뿐 아니라 cycle도 반드시 같이 맞추는 것이 안전합니다.
- exact current / exact field / finite exact validation은 가능한 한 사용한 LUT 조건과 같은 주파수/파형/사이클/레벨로 저장하는 것이 가장 안전합니다.
