# UI Polish Report

- 생성 시각: 2026-04-16
- 범위: `app_ui.py` 기준 UI / 실사용 polish / AppTest 대체 validation / 실행 편의성
- 제외: `validation_retune.py` 핵심 구조 재설계, artifact generator 구조 변경, 운영 정책 변경

## 이번 패스에서 반영한 내용

- Quick LUT 상단에 `빠른 시작 예제` 카드형 진입점을 추가했습니다.
- 요청 조건, 지원 상태, 엔진, 다음 행동을 한 번에 보는 `요청 상태 한눈에 보기` 카드 블록을 추가했습니다.
- exact / provisional / preview-only / unsupported 판정을 `라우팅 판정과 다음 행동` 블록으로 분리했습니다.
- unsupported / preview-only에서도 nearest exact, nearest provisional, 추가 측정 추천이 바로 보이도록 정리했습니다.
- `Validation / Retune 메인 흐름` 안내 블록을 exact 결과 흐름 앞쪽에 배치했습니다.
- `Exact Matrix / Provisional / ROI`를 Continuous / Finite / Missing / ROI 관점으로 다시 묶었습니다.
- 문서 패키지 읽기 순서에 `ui_guide.md`를 추가했습니다.

## 검증 결과

- `python D:\programs\Codex\Coil Analyzing\tools\validate_exact_matrix_ui_states.py`
  - 결과: pass
  - 산출물: `artifacts/policy_eval/exact_matrix_ui_validation.json`
- `python D:\programs\Codex\Coil Analyzing\tools\validate_request_router_actions.py`
  - 결과: pass
  - 산출물: `artifacts/policy_eval/request_router_validation.json`
- `python -m py_compile D:\programs\Codex\src\field_analysis\app_ui.py`
  - 결과: pass

## 바탕화면 실행 편의성

- 현재 준비된 바로가기
- `C:\Users\jaech\Desktop\전자기장 LUT 보정 툴.lnk`
- `C:\Users\jaech\Desktop\전자기장 LUT 보정 툴 (실사용 모드).lnk`
- `C:\Users\jaech\Desktop\예제 1 연속 Bz exact.lnk`
- `C:\Users\jaech\Desktop\예제 2 연속 current exact.lnk`
- `C:\Users\jaech\Desktop\예제 3 finite triangle exact.lnk`

## 남은 비차단 항목

- headless validation 스크립트 실행 시 Streamlit bare-mode cache warning이 출력됩니다.
- 이 경고는 UI deprecation warning과 달리 기능 실패는 아니며, JSON/Markdown 산출물 생성은 정상 완료됩니다.
- 코어 validation 관련 pytest 실패는 현재 UI 쓰레드 범위를 벗어난 항목으로 남겨두었습니다.
