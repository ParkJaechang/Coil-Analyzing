# 전자기장 LUT/보정 운용 툴

`D:\programs\Codex\Coil Analyzing\app_field_analysis_quick.py` 는 `<= 5 Hz` exact-supported 운용 범위를 빠르게 확인하고, LUT / 제어 수식 / 보정 파형을 내보내는 Streamlit 앱입니다.

## 실행 방법

가장 쉬운 방법:

- 일반 실행: [launch_quick_lut.cmd](D:\programs\Codex\Coil Analyzing\launch_quick_lut.cmd)
- 실사용 모드 기본 실행: [launch_quick_lut_operational.cmd](D:\programs\Codex\Coil Analyzing\launch_quick_lut_operational.cmd)

직접 실행:

```powershell
& "D:\programs\Codex\.venv\Scripts\streamlit.exe" run "D:\programs\Codex\Coil Analyzing\app_field_analysis_quick.py"
```

## 공식 지원 범위

- 공식 운영 대역: `0.25 ~ 5 Hz`
- 자동 추천 운영:
  - `continuous / current / exact support / <= 5 Hz`
- software-ready:
  - `continuous / field / exact support / <= 5 Hz`
- finite exact-supported:
  - `<= 5 Hz`
  - `sine 47 + triangle 48 = 95 recipes`
- provisional preview:
  - `sine / 1.0 Hz / 1.0 cycle / 20 pp`
  - 근거: `1.0 Hz / 1.0 cycle / 10 pp exact`를 `2x` 스케일
- preview-only:
  - `interpolated_in_hull`
  - finite exact recipe 표 밖 요청
- blocked 또는 reference-only:
  - `interpolated_edge`
  - `out_of_hull`
  - `> 5 Hz`

## 상태 의미

- `exact`: 공식 exact-supported 경로. 자동 추천과 다운로드 허용.
- `provisional`: exact가 없는 한 칸을 임시 대체 조합으로 제공. 실험 모드 전용.
- `preview-only`: 계산 결과는 참고용. 자동 다운로드 금지.
- `unsupported`: 공식 지원 범위 밖. nearest exact/provisional 및 추가 측정 추천 확인.

## 사용 흐름

1. 좌측에서 연속/finite/검증/LCR 파일을 업로드하거나 기억된 데이터를 재사용합니다.
2. `Quick LUT` 화면에서 파형, 주파수, 목표 타입(current/field), 레벨, cycle을 고릅니다.
3. 상단 `요청 라우터`에서 현재 상태가 `exact / provisional / preview-only / unsupported` 중 무엇인지 확인합니다.
4. 필요하면 `가장 가까운 exact 조합으로 전환` 또는 `가장 가까운 provisional 조합으로 미리보기`를 사용합니다.
5. 결과 화면에서 추천 전압, 예상 출력, 엔진/신뢰도, export 파일 접두사를 확인합니다.
6. `제어 LUT CSV`, `제어 수식 TXT`, `보정 전압 파형 CSV`를 내려받습니다.

## 대표 시나리오

- 연속 전류 exact:
  - `sine / 0.5 Hz / current 20 A`
- 연속 자기장 exact:
  - `sine / 0.25 Hz / field 20 mT`
- finite triangle exact:
  - `triangle / 1.0 Hz / 1.0 cycle / 20 pp`

앱 상단의 `대표 예제 바로 불러오기` 버튼으로 바로 세팅할 수 있습니다.

## 업로드 폴더와 기억된 데이터

앱은 아래 캐시 폴더를 자동으로 읽고, 새 파일이 들어오면 scope/report를 자동 갱신합니다.

- continuous: [D:\programs\Codex\outputs\field_analysis_app_state\uploads\continuous](D:\programs\Codex\outputs\field_analysis_app_state\uploads\continuous)
- transient: [D:\programs\Codex\outputs\field_analysis_app_state\uploads\transient](D:\programs\Codex\outputs\field_analysis_app_state\uploads\transient)
- validation: [D:\programs\Codex\outputs\field_analysis_app_state\uploads\validation](D:\programs\Codex\outputs\field_analysis_app_state\uploads\validation)
- lcr: [D:\programs\Codex\outputs\field_analysis_app_state\uploads\lcr](D:\programs\Codex\outputs\field_analysis_app_state\uploads\lcr)

## export 파일

대표 export는 아래 4종입니다.

- `*_control_lut.csv`
- `*_formula.txt`
- `*_coefficients.csv`
- `*.csv` recommended command waveform

파일명에는 waveform, freq, target type, level, cycle 조건이 포함됩니다.

## known limitations

- interpolated auto는 닫혀 있습니다.
- finite generalization은 preview-only입니다.
- `continuous / field exact`는 software-ready 상태이며, 공식 승격에는 외부 bench sign-off가 남아 있습니다.
- 남은 exact 실측 공백은 `sine / 1.0 Hz / 1.0 cycle / 20 pp` 1개입니다.

## 테스트/검증

PC 안에서 닫힌 주요 검증:

- exact/provisional/preview/unsupported UI regression
- request router one-click transition validation
- exact export 파일 생성/헤더/행 수/메타 검증
- finite triangle exact software path validation
- scope auto-refresh validation

대표 artifact:

- [exact_matrix_closeout.md](D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\exact_matrix_closeout.md)
- [exact_export_validation.md](D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\exact_export_validation.md)
- [release_preflight.md](D:\programs\Codex\Coil Analyzing\artifacts\policy_eval\release_preflight.md)

## 외부 작업으로 남아 있는 항목

- bench sign-off
- 남은 exact 실측 1개 추가
