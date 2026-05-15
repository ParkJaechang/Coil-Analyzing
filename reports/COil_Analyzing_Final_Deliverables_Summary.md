# 최종 산출물 요약

## 1. 생성한 산출물

| 산출물 | 파일 또는 링크 | 용도 |
|---|---|---|
| DOCX 정식 보고서 | reports/COil_Analyzing_Development_Report.docx | 제출/공유 가능한 문서형 정식 보고자료 |
| DOCX 보고서 PDF 변환본 | 생성하지 못함 | Word COM 자동 변환이 시간 초과되어 미생성 |
| Canva 시각형 보고서 링크 | https://www.canva.com/d/bvaDXs8dwfAFiBA | 12페이지 시각형 보고자료 |
| Canva 편집 링크 | https://www.canva.com/d/-mF3UfqDMPRn9OY | Canva 내 최종 검토 및 수정용 |
| Canva 보조 Doc 링크 | https://www.canva.com/d/qdP0fa0Hse7vcg2 | 원고 보존용 Canva Doc |
| Canva PDF export | 생성하지 못함 | 현재 Canva 도구에서 export 기능 미노출 |
| Canva 제작 결과 요약 | reports/COil_Analyzing_Canva_Result.md | Canva 생성 링크와 확인 필요 사항 정리 |
| 최종 산출물 요약 | reports/COil_Analyzing_Final_Deliverables_Summary.md | 생성 파일, 원본 자료, 사용 방법 정리 |

## 2. 원본 자료

| 원본 파일 | 사용 목적 |
|---|---|
| docs/coil_analyzing_docs_report_final.md | DOCX 정식 보고서 본문 기준 |
| docs/coil_analyzing_canva_report_final.md | Canva 시각형 보고서 구성 기준 |
| docs/coil_analyzing_report_revision_summary.md | 최종 수정 방향과 확인 필요 사항 참고 |

## 3. 변환 과정에서 적용한 사항

- Markdown 제목을 DOCX 제목 스타일로 변환했다.
- 표지 페이지, 목차, 본문, 근거 섹션이 구분되도록 DOCX 구조를 정리했다.
- Markdown 표를 Word 표 형식으로 변환했다.
- Mermaid 코드블록은 raw 코드로 두지 않고 DOCX 내 단계형 도식 표로 재구성했다.
- 코드블록, raw Markdown 표기, Canva 전용 표현이 DOCX 본문에 남지 않도록 정리했다.
- 한글 보고서에 맞춰 맑은 고딕 계열 글꼴을 적용했다.
- 바닥글에 문서명과 페이지 번호 필드를 추가했다.
- Canva에는 `docs/coil_analyzing_canva_report_final.md`의 12페이지 구성을 기준으로 실제 디자인 생성을 시도했다.
- Docs용 보고서와 Canva용 보고서 모두 개발 목적, 사용 전략, 알고리즘, 데이터 흐름, 최신 진행사항, 결과 해석 중심을 유지했다.

## 4. 남아 있는 확인 필요 사항

- 실제 코일 배선, 센서 배치, 장비 연결 구조 같은 물리 토폴로지는 저장소 코드만으로는 일부 확인이 필요하다.
- Raw Waveforms와 Quick LUT 실제 화면 캡처가 추가되면 Canva 보고서의 시각 완성도를 높일 수 있다.
- 실제 결과 그래프가 추가되면 target/support/predicted/command 관계 설명을 더 명확하게 만들 수 있다.
- Canva 자동 생성 결과에는 일부 템플릿 placeholder 또는 일반 보고서 템플릿 문구가 남을 수 있으므로, 제출 전 Canva 내 수동 검토가 필요하다.
- 현재 도구 환경에서는 Canva PDF export 기능이 노출되지 않아 Canva PDF 파일은 생성하지 못했다.
- Word COM 기반 PDF 변환은 시간 초과되어 DOCX 보고서 PDF 파일은 생성하지 못했다. 로컬 Word에서 `파일 > 내보내기 > PDF`로 변환하는 방식이 필요하다.

## 5. 사용 방법

- DOCX 보고서는 Word, Google Docs, PDF 변환용 원문 보고자료로 사용한다.
- Canva 시각형 보고서는 Docs용 보고서의 핵심 내용을 시각적으로 훑어보는 보고자료로 사용한다.
- Canva 제출 전에는 편집 링크에서 각 페이지의 placeholder와 문구 누락 여부를 확인한다.
- PDF 제출이 필요하면 DOCX는 Word 또는 Google Docs에서 PDF로 내보내고, Canva 보고서는 Canva의 공유/다운로드 기능으로 PDF를 내보낸다.
