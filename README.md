# Coil Large-Signal Analyzer

로컬 CSV/XLSX 시계열 데이터와 워크스페이스 reference 파일을 이용해 코일 / 전자석의 large-signal 전기적 특성과 자기장 응답을 분석하는 Streamlit 앱이다.

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

앱은 빈 상태에서도 실행되며, 업로드할 파일이 없으면 안내 화면과 built-in example dataset를 제공한다.

## 지원 파일 형식

- 입력 waveform: `CSV`, `XLSX`
- 테스트 요청표: `CSV`, `XLSX`
- reference workbook: `all_bands_full.xlsx`
- 참고 문서: `7224-Datasheet-05-06-24 (1).pdf`, `7224-7226_OperatorManual-1.pdf`, `코스모크 전자석_Silicon steel (25.10.28).pdf`

파일이 실제로 존재하지 않아도 앱은 종료되지 않는다. workbook / PDF 상태는 Home 페이지에서 확인할 수 있다.

## 채널 매핑 방법

1. `Data Import` 에서 waveform 파일을 올린다.
2. `Channel Mapping & Calibration` 에서 시간, 전압, 전류, 자기장 채널을 선택한다.
3. 각 채널에 대해 아래 항목을 설정한다.
   - scale
   - offset
   - polarity invert
   - delay correction
   - unit
4. time 단위는 `s`, `ms`, `us`, `datetime` 을 지원한다.
5. timestamp가 `2026-04-07T16:45:31.5746008+09:00` 같은 ISO datetime 문자열이어도 분석 가능하다.
6. 자기장 채널은 여러 개를 등록할 수 있고 alias 를 지정할 수 있다.
7. `전류 기준 자동 정렬` 을 누르면 cross-correlation 기반 delay 를 current 채널 기준으로 추정한다.

한 번 저장한 컬럼 패턴은 workspace-local mapping library에 저장되어 유사한 파일에 재사용된다.

## 계산식 설명

기본파 추정은 zero-crossing 단독 방식이 아니라 sine fit / single-frequency basis fit 기반이다.

- `|Z1| = V1_pk / I1_pk`
- `delta_phi_VI = phase_V - phase_I`
- `Req = |Z1| * cos(delta_phi_VI)`
- `Xeq = |Z1| * sin(delta_phi_VI)`
- `Leq = Xeq / (2*pi*f)`
- `K_BI = B1_pk / I1_pk`
- `K_BV = B1_pk / V1_pk`
- `alpha = Vout_pk / (Vin_pk * G_mode)`
- `lambda(t) = integral(v(t) - Rdc*i(t)) dt`

추가 지표:

- raw peak-to-peak
- raw RMS
- fundamental peak
- fundamental RMS
- phase in degree
- phase delay in seconds
- crest factor
- THD, if data quality permits

## Large-Signal vs Small-Signal

- `Electrical Analysis`, `Magnetic Analysis`, `Gain / Drive Requirement Analysis` 는 waveform 기반 measured large-signal 결과를 사용한다.
- `LCR Reference Comparison` 은 `all_bands_full.xlsx` 의 small-signal reference 를 별도로 읽어 비교한다.
- 둘은 같은 표나 계산에서 자동으로 섞이지 않는다.
- LCR meter series R 은 자동으로 copper resistance 또는 `Rdc` 로 간주하지 않는다.
- `Rdc(T)` 는 `Advanced Analysis` 에서 별도 입력값으로 취급한다.

## 예시 워크플로우

1. reference file 상태 확인
2. waveform CSV/XLSX 업로드
3. 채널 매핑
4. calibration factor / polarity / unit / delay 입력
5. 테스트 요청 포인트와 실제 데이터 매칭
6. cycle start / cycle count 기반 analysis window 선택
7. Signal Analysis 실행
8. Electrical / Magnetic / Advanced / Gain / Reference 페이지 확인
9. Export 페이지에서 Excel + HTML + JSON bundle 다운로드

## 사용 편의 개선 사항

- 사이드바를 단계형 한국어 UI로 정리했다.
- 각 페이지 상단에 `빠른 가이드` 를 넣어, 지금 해야 할 작업을 바로 볼 수 있다.
- Gain 분석 페이지에서는 실측 `Vout_pk` 입력값이 있으면 그것을 우선 사용한다.
- `Vout_pk` 가 없으면 waveform에서 계산된 `V1_pk` 를 사용하고, 그것도 없으면 measured `|Z1|` 와 목표 `Ipp` 로 required `Vout_pk` 를 역산한다.

## UI 페이지 개요

- `Home / Project Overview`: 프로젝트 상태, reference 상태, 최근 업로드, 누락 포인트
- `Data Import`: waveform 업로드, preview, sheet 선택, metadata 입력
- `Test Request Manager`: preset 요청표, 요청표 업로드, 상태 테이블
- `Channel Mapping & Calibration`: 채널 지정, scale/offset/invert/delay, auto alignment
- `Signal Analysis`: 정수 주기 window, detrend, offset removal, zero-phase smoothing, fitted waveform
- `Electrical Analysis`: large-signal impedance / power / phase
- `Magnetic Analysis`: B amplitude / phase / B-I / B-V
- `Advanced Analysis`: lambda-i, differential inductance 추정
- `Gain / Drive Requirement Analysis`: alpha, gain mode 20 V/V or 6 V/V, overload 판단
- `LCR Reference Comparison`: `all_bands_full.xlsx` multi-sheet selector, measured vs reference plot
- `Export`: Excel, HTML plot, JSON settings bundle

## 알려진 한계

- 1차 버전의 중심은 sine-like waveform 분석이다.
- 긴 raw 파일을 자동 세그먼트로 쪼개는 기능은 아직 없다. 현재는 파일 단위 + cycle window 선택 중심이다.
- PDF 본문 파싱은 하지 않는다. 현재는 파일 존재 상태와 workbook 중심 fallback 구조다.
- multi-sheet waveform workbook 은 파일당 선택 sheet 하나를 기준으로 읽는다.
- cross-correlation alignment 는 충분한 excitation 이 있어야 안정적이다.
- THD 는 간단한 harmonic magnitude 기반이므로 window quality에 민감하다.

## 테스트

```bash
pytest
```

## 파일 구조

```text
app.py
config/example_settings.json
src/coil_analyzer/
tests/
requirements.txt
README.md
```
