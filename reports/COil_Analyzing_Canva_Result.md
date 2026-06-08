# COil Analyzing Canva 시각형 보고서 생성 결과

## 1. Canva 디자인 결과

| 항목 | 결과 |
|---|---|
| 기본 Canva 시각형 보고서 | https://www.canva.com/d/bvaDXs8dwfAFiBA |
| 편집 URL | https://www.canva.com/d/-mF3UfqDMPRn9OY |
| Canva 디자인 ID | DAHJuV_W-2c |
| 페이지 수 | 12페이지 |
| 디자인 유형 | Canva report |

## 2. 보조 Canva Doc 결과

Canva report 템플릿 생성 과정에서 일부 페이지에 템플릿 placeholder가 남는 문제가 확인되어, 원고 보존 목적의 Canva Doc도 추가 생성했다.

| 항목 | 결과 |
|---|---|
| 보조 Canva Doc | https://www.canva.com/d/qdP0fa0Hse7vcg2 |
| 편집 URL | https://www.canva.com/d/N7pBUiCgKMEKW3w |
| Canva 디자인 ID | DAHJuekmVBg |
| 페이지 수 | 1개 Doc |
| 디자인 유형 | Canva Doc |

## 3. 생성 기준

원본 자료는 다음 파일을 기준으로 사용했다.

| 원본 파일 | 사용 목적 |
|---|---|
| docs/coil_analyzing_docs_report_final.md | Docs용 정식 보고서의 내용 기준 |
| docs/coil_analyzing_canva_report_final.md | Canva 시각형 보고서의 페이지 구성 기준 |
| docs/coil_analyzing_report_revision_summary.md | 수정 방향과 확인 필요 사항 참고 |

## 4. Canva 페이지 구성 요약

| 페이지 | 제목 | 핵심 내용 |
|---|---|---|
| 1 | 표지 | COil Analyzing 개발 목적 및 구현 전략 보고서 |
| 2 | 왜 이 프로젝트가 필요한가 | DAQ command 역추정 문제와 수작업 분석 한계 |
| 3 | 프로젝트의 핵심 목적 | 데이터 정리, 품질 검수, command 후보 계산, 결과 해석 |
| 4 | 전체 분석 흐름 | 측정 파일에서 predicted output까지 이어지는 처리 흐름 |
| 5 | 사용 전략 | 입력, 전처리, 지표 추출, 조건 비교, 결과 해석 |
| 6 | 핵심 알고리즘/분석 로직 | metadata parsing, baseline correction, finite-cycle coverage, support selection |
| 7 | 데이터가 결과로 바뀌는 과정 | 원본 측정 데이터에서 command 후보와 예상 출력으로 변환 |
| 8 | 개발 과정에서 개선된 점 | 데이터 처리 안정성, 의미 분리, Raw Waveforms/Quick LUT 해석 구조 |
| 9 | 최신 진행사항 | field-only, rounded triangle 100pp, support reference, finite-cycle 검토 상태 |
| 10 | 한계와 향후 개선 방향 | 데이터 검수, support coverage, target/predicted 비교, 모델링 안정화 |
| 11 | 비전공자용 요약 | 무엇을 분석하고 왜 필요한지 한 장으로 요약 |
| 12 | 근거 및 참고 | Markdown 원고, 코드, 문서, 테스트, 변경 이력 근거 |

## 5. 확인 필요 사항

- Canva report 자동 생성 결과에는 일부 템플릿 placeholder 또는 일반 보고서 템플릿 문구가 남을 수 있으므로 Canva 내 최종 수동 검토가 필요하다.
- 현재 도구 환경에서는 Canva PDF export 기능이 노출되지 않아 `reports/COil_Analyzing_Canva_Visual_Report.pdf`는 생성하지 못했다.
- 실제 제출 전에는 Canva에서 각 페이지의 텍스트가 원문 의도와 일치하는지 확인하고, placeholder가 남아 있으면 삭제해야 한다.
