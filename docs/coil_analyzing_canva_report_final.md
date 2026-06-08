# COil Analyzing Canva용 시각 보고서 최종 원고

이 문서는 Canva에서 바로 옮겨 제작할 수 있는 시각형 보고자료 원고이다. 발표자료가 아니라, 읽는 사람이 페이지를 넘기며 프로젝트 목적, 분석 흐름, 알고리즘, 최신 상태를 이해할 수 있도록 구성한다.

## 전체 디자인 지침

- 용도: 시각적 보고자료
- 페이지 수: 12페이지
- 톤: 깔끔한 엔지니어링 보고서
- 배경: 흰색 또는 아주 연한 회색
- 주요 색상: 짙은 남색, 청록색, 회색, 주의 영역에만 노란색/빨간색
- 구성: 카드형 요약, 흐름도, 비교표, 체크리스트 중심
- 금지: 과장된 홍보 문구, 발표 대본, speaker notes, PR/커밋 중심 설명

---

## Page 1. 표지

**핵심 메시지:**  
COil Analyzing은 코일 측정 데이터를 정리하고, 목표 자기장 파형에 필요한 DAQ command 후보를 계산하는 분석 프로젝트이다.

**본문 문구:**  
- COil Analyzing 개발 목적 및 구현 전략 보고서
- 코일 분석을 위한 데이터 처리, 분석 로직, 최신 구현 상태 정리
- 측정 데이터 기반 command 추천과 결과 해석 흐름

**시각 요소:**  
코일/DAQ 아이콘 → 데이터 → 분석 → command/예상 출력 흐름 라인 다이어그램

**디자인 지시:**  
상단에 제목, 중앙에 4단 라인 다이어그램, 하단에 키워드 4개 배치.

**원문 보고서 근거:**  
Docs용 보고서 1장, 2장

**Canva에 넣을 최종 문구:**  
```text
COil Analyzing 개발 목적 및 구현 전략 보고서
코일 분석을 위한 데이터 처리, 분석 로직, 최신 구현 상태 정리

측정 데이터 → 검수 → 분석 로직 → 추천 command 및 예상 출력
```

---

## Page 2. 왜 이 프로젝트가 필요한가

**핵심 메시지:**  
원하는 자기장 파형을 만들려면, 입력 전압과 실제 자기장 출력 사이의 관계를 데이터 기반으로 이해해야 한다.

**본문 문구:**  
- 수작업 분석은 파일 조건, 컬럼, 그래프 의미를 매번 사람이 판단해야 한다.
- 노이즈, baseline, clipping, finite-cycle tail 문제를 놓치기 쉽다.
- 프로그램은 같은 기준으로 데이터를 정리하고 결과를 비교하게 해준다.

**시각 요소:**  
Before / After 비교 카드

**디자인 지시:**  
왼쪽은 “수작업 분석”, 오른쪽은 “프로그램 기반 분석”으로 구성. 왼쪽은 회색/노란색, 오른쪽은 청록색 강조.

**원문 보고서 근거:**  
Docs용 보고서 2장

**Canva에 넣을 최종 문구:**  
```text
수작업 분석
- 파일 조건과 컬럼을 매번 확인
- raw/corrected 차이 해석 어려움
- target/support/predicted/command 혼동 가능
- finite-cycle 종료부와 tail 판단 어려움

프로그램 기반 분석
- 데이터 조건을 일관되게 정리
- Raw Waveforms에서 품질 검수
- Quick LUT에서 결과 의미 분리
- support 기반 command 후보 계산
```

---

## Page 3. 프로젝트의 핵심 목적

**핵심 메시지:**  
목표 자기장 파형을 만들기 위해 필요한 DAQ 입력 전압 command 후보를 측정 데이터 기반으로 계산한다.

**본문 문구:**  
- 입력 데이터 정리
- 데이터 품질 검수
- command 후보 계산
- 결과 해석

**시각 요소:**  
2x2 목적 카드

**디자인 지시:**  
각 카드에 짧은 제목과 1문장 설명. 아이콘은 파일, 체크, 계산, 그래프 사용.

**원문 보고서 근거:**  
Docs용 보고서 2장, 3장

**Canva에 넣을 최종 문구:**  
```text
입력 데이터 정리
측정 파일과 metadata를 표준 형식으로 변환

데이터 품질 검수
raw/corrected waveform과 anomaly flag 확인

command 후보 계산
support reference 기반 DAQ 입력 전압 후보 계산

결과 해석
Physical Target, Support Reference, Predicted Output, Recommended Command 분리 비교
```

---

## Page 4. 전체 분석 흐름

**핵심 메시지:**  
데이터는 입력, 정리, 특징 추출, 분석, 결과 해석 단계를 거쳐 의미 있는 정보로 변환된다.

**본문 문구:**  
측정 파일 입력 → Parser → Preprocessing → Cycle & Metrics → Support Reference → Recommended Command / Predicted Output

**시각 요소:**  
가로형 6단계 플로우차트

**디자인 지시:**  
각 단계는 박스로 만들고 화살표 연결. `Raw Waveforms 검수`와 `Quick LUT 결과 해석`은 보조 라벨로 표시.

**원문 보고서 근거:**  
Docs용 보고서 5장, 7장

**Canva에 넣을 최종 문구:**  
```text
측정 파일 입력
→ Parser / metadata 해석
→ Preprocessing / 데이터 보정
→ Cycle & Metrics / 특징 추출
→ Support Reference 구성
→ Recommended Command + Predicted Output 확인

Raw Waveforms = 데이터 품질 검수
Quick LUT = 추천 command와 예상 출력 해석
```

---

## Page 5. 사용 전략

**핵심 메시지:**  
복잡한 측정 데이터를 단계별로 나누어 처리하고, 의미 있는 값만 결과 해석에 연결한다.

**본문 문구:**  
- 입력 전략: 파일명과 metadata로 조건 식별
- 정리 전략: baseline, sign, smoothing, outlier 처리
- 분석 전략: cycle, PP, drift, support 관계 계산
- 해석 전략: target/support/predicted/command 분리

**시각 요소:**  
4개 전략 카드

**디자인 지시:**  
카드마다 “목적 / 결과”를 1줄씩 넣는다.

**원문 보고서 근거:**  
Docs용 보고서 5장

**Canva에 넣을 최종 문구:**  
```text
입력 전략
파일명과 metadata로 측정 조건 식별

정리 전략
baseline, sign, smoothing, outlier 처리

분석 전략
cycle, PP, drift, support 관계 계산

해석 전략
target/support/predicted/command 분리 비교
```

---

## Page 6. 핵심 알고리즘/분석 로직

**핵심 메시지:**  
알고리즘은 데이터를 단순 표시하는 것이 아니라, 의미 있는 패턴과 판단 기준을 찾아내는 역할을 한다.

**본문 문구:**  
5개 핵심 로직을 짧게 설명한다.

**시각 요소:**  
알고리즘 카드 5개

**디자인 지시:**  
각 카드에 “로직 이름 / 하는 일 / 쉬운 비유 / 결과 의미”를 배치.

**원문 보고서 근거:**  
Docs용 보고서 6장

**Canva에 넣을 최종 문구:**  
```text
Metadata 추론
파일명 라벨을 읽어 측정 조건을 식별

Baseline 보정
저울 0점 맞추기처럼 기준선을 정렬

Cycle Detection
긴 신호에서 반복 마디를 찾음

LUT 보간
측정값 사이에서 필요한 전압 크기 추정

Harmonic Inverse
파형을 성분별로 나누어 필요한 입력 전압을 역산
```

---

## Page 7. 데이터가 결과로 바뀌는 과정

**핵심 메시지:**  
원본 데이터는 여러 처리 단계를 거치며 해석 가능한 결과값으로 변환된다.

**본문 문구:**  
- 원본 데이터: time, voltage, current, magnetic field
- 중간 처리값: corrected waveform, cycle index, PP, drift
- 분석 기준: field-only, rounded triangle, 100pp, support reference
- 최종 결과: Recommended Command, Predicted Output, finite metrics

**시각 요소:**  
데이터 변환 파이프라인

**디자인 지시:**  
왼쪽에서 오른쪽으로 갈수록 색상을 진하게 하여 정보가 정제되는 느낌을 표현.

**원문 보고서 근거:**  
Docs용 보고서 5장, 9장

**Canva에 넣을 최종 문구:**  
```text
원본 데이터
time, voltage, current, magnetic field

중간 처리값
corrected waveform, cycle index, PP, drift

분석 기준
field-only, rounded triangle, 100pp, support reference

최종 결과
Recommended Command, Predicted Output, finite metrics
```

---

## Page 8. 개발 과정에서 개선된 점

**핵심 메시지:**  
개발 과정은 단순 기능 추가가 아니라, 분석 안정성과 해석 가능성을 높이는 방향으로 진행되었다.

**본문 문구:**  
개선 전/후 비교

**시각 요소:**  
4행 비교표

**디자인 지시:**  
왼쪽은 기존 문제, 오른쪽은 개선 방향. 개선 방향은 청록색으로 강조.

**원문 보고서 근거:**  
Docs용 보고서 10장

**Canva에 넣을 최종 문구:**  
```text
데이터 처리 안정성
조건/컬럼 혼동 → 파일명 metadata와 Raw Waveforms 검수

분석 모델링 방향
전류/장비 조건 혼선 → field-only 목표 기준 정리

결과 해석 방식
그래프 의미 혼동 → target/support/predicted/command 분리

finite-cycle 해석
종료부 문제 누락 → active/terminal/tail 지표 분리
```

---

## Page 9. 최신 진행사항

**핵심 메시지:**  
최신 상태에서는 입력 데이터 처리부터 추천 command 계산, 결과 해석, feedback 확장까지 핵심 흐름이 정리되고 있다.

**본문 문구:**  
구현됨 / 개발 중 / 검증 필요

**시각 요소:**  
3열 상태 체크리스트

**디자인 지시:**  
구현됨은 체크 아이콘, 개발 중은 진행 아이콘, 검증 필요는 주의 아이콘 사용.

**원문 보고서 근거:**  
Docs용 보고서 8장

**Canva에 넣을 최종 문구:**  
```text
구현됨
- 측정 파일 입력 및 metadata 추론
- Raw Waveforms 검수
- Quick LUT command 후보 계산
- hardware limit 및 finite metrics 표시

개발 중
- actual-drive review
- feedback correction
- second modeled voltage LUT
- continuous steady-state 1 cycle 추출

검증 필요
- finite-cycle exact support
- feedback 결과 실측 검증
- final LUT export/review 운용 절차
```

---

## Page 10. 한계와 향후 개선 방향

**핵심 메시지:**  
현재 구현은 기본 분석 흐름을 갖추었지만, 데이터 품질과 실측 검증 측면에서 추가 개선이 필요하다.

**본문 문구:**  
Now / Next / Future 구조

**시각 요소:**  
3단 컬럼

**디자인 지시:**  
Now는 현재 상태, Next는 가까운 개선, Future는 장기 확장으로 배치.

**원문 보고서 근거:**  
Docs용 보고서 11장

**Canva에 넣을 최종 문구:**  
```text
Now
측정 데이터 기반 command 후보 계산과 결과 해석 구조 구축

Next
Raw Waveforms 검수 기준, exact support matrix, feedback 검증 강화

Future
실험 결과 기반 feedback 보정과 실무 보고/export 자동화

주의
Recommended Command는 정답이 아니라 측정 데이터 기반 후보이다.
```

---

## Page 11. 비전공자용 한 장 요약

**핵심 메시지:**  
COil Analyzing은 코일 데이터를 사람이 이해할 수 있는 분석 결과로 바꾸는 도구이다.

**본문 문구:**  
무엇 / 왜 / 어떻게 / 결과

**시각 요소:**  
4단 요약 카드

**디자인 지시:**  
각 카드의 텍스트는 2줄 이내. 아이콘을 크게 사용.

**원문 보고서 근거:**  
Docs용 보고서 12장

**Canva에 넣을 최종 문구:**  
```text
무엇을 분석하는가
DAQ 전압, 코일 전류, 자기장 측정 데이터

왜 필요한가
원하는 자기장을 만들 전압 command는 직접 알기 어렵기 때문

어떻게 분석하는가
데이터를 정리하고 support reference와 목표 파형을 비교

결과가 의미하는 것
DAQ 입력 전압 후보와 예상 자기장 출력
```

---

## Page 12. 근거 및 참고

**핵심 메시지:**  
본 보고서는 저장소의 코드, 문서, 테스트, 변경 이력을 기반으로 작성되었다.

**본문 문구:**  
근거 자료 유형별 요약

**시각 요소:**  
근거 자료 표

**디자인 지시:**  
작은 글씨로 배치하되, 본문과 구분되는 회색 박스 사용. PR/커밋은 강조하지 않는다.

**원문 보고서 근거:**  
Docs용 보고서 13장

**Canva에 넣을 최종 문구:**  
```text
주요 코드 근거
parser.py, preprocessing.py, cycle_detection.py, metrics.py,
lut.py, compensation.py, hardware.py, finite_cycle_metrics.py

주요 문서 근거
README.md, 모델링_정책.md, 데이터_수집_가이드.md,
Raw_Waveforms_데이터_검수_가이드.md

주요 테스트 근거
field-only Quick LUT, finite-cycle metrics,
continuous steady-state extraction, target semantics tests

변경 이력 근거
원격 main 기준 및 최신 로컬 개발 브랜치 확인
```
