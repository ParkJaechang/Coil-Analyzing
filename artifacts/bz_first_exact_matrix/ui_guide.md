# UI Guide

- 이 문서는 앱을 처음 켠 사용자가 어느 화면부터 보고, 어떤 버튼을 눌러야 하는지 설명합니다.
- 핵심은 `Quick LUT` 화면에서 요청 상태를 먼저 확인하고, exact인 경우 바로 `Validation / Retune`로 이어가는 것입니다.
- exact / provisional / preview-only / unsupported는 색상 배지와 상태 카드로 구분됩니다.
- 지원되지 않는 요청도 끝나는 것이 아니라, nearest exact / provisional / 추가 측정 추천으로 다음 행동을 안내합니다.

## 1. 처음 열었을 때

- 상단 `빠른 시작 예제`에서 대표 시나리오 3개를 바로 불러올 수 있습니다.
- 연속 Bz exact, 연속 current exact, finite triangle exact를 각각 원클릭으로 체험할 수 있습니다.
- 실사용에서는 파일을 업로드하지 않아도 기억된 업로드 폴더와 운영 scope 요약을 먼저 확인할 수 있습니다.

## 2. 요청 상태 한눈에 보기

- `현재 요청`: 지금 입력한 파형, 주파수, 타깃 축, 목표 레벨을 보여줍니다.
- `지원 상태 / route`: 공식 exact, provisional 실험, preview-only, 지원 불가 중 어디인지 즉시 표시합니다.
- `엔진 / 운영 모드`: Exact Engine, Provisional Engine, Preview Engine, Route Selector 중 어떤 경로가 선택됐는지 보여줍니다.
- `다음 행동`: 바로 다운로드 가능한지, validation이 필요한지, nearest exact로 옮겨야 하는지 안내합니다.

## 3. 라우팅 판정과 다음 행동

- exact: 현재 조건이 공식 운영 범위 안입니다.
- provisional: 정식 exact가 아닌 실험용 임시 경로입니다.
- preview-only: 계산 미리보기까지만 허용되고 운영 export는 막혀 있습니다.
- unsupported: 공식 지원 범위를 벗어나 nearest exact / 추가 측정 추천을 먼저 봐야 합니다.

## 4. Validation / Retune 메인 흐름

- 1단계: 현재 추천 또는 과거 recommendation을 선택합니다.
- 2단계: validation run을 업로드하면 앱이 자동으로 매칭 후보를 정리합니다.
- 3단계: target / predicted / actual / corrected를 비교해서 품질 배지를 확인합니다.
- 4단계: corrected waveform, corrected LUT, report를 다운로드합니다.

## 5. Exact Matrix / ROI 보기

- `Continuous` 탭: continuous/current exact auto와 continuous/field software-ready 범위를 봅니다.
- `Finite` 탭: sine 47 + triangle 48, 총 95개 exact recipe를 확인합니다.
- `Missing / Provisional` 탭: 남은 exact 공백 1칸과 provisional 셀을 분리해서 보여줍니다.
- `ROI / Reference` 탭: 추가 측정 우선순위와 5 Hz 초과 reference-only 셀을 확인합니다.

## 6. 실사용 팁

- Bz-first가 기본이므로, 먼저 `field(Bz)`로 목표 파형을 맞추고 current는 제약/진단으로 보세요.
- field exact는 software-ready 상태이므로 bench sign-off 전까지는 validation을 권장합니다.
- unsupported 요청은 실패가 아니라 측정 계획이 필요한 상태입니다.
- corrected LUT를 새 기준으로 삼을 때는 catalog와 quality badge를 함께 확인하세요.
