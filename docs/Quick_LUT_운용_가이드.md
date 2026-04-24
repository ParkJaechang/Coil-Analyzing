# Quick LUT 운용 가이드

## 현재 운용 철학

Quick LUT는 field-first 운용 화면입니다.

- target field shape: rounded triangle fixed
- target field pp: 100pp fixed
- support/input waveform family: 실험 support family 선택용
- current/gain/hardware/LCR: main shape selection 근거가 아님

## 실행

```powershell
.\launch_quick_lut_local.cmd
```

전체 앱에서 Quick LUT를 보려면:

```powershell
.\launch_field_analysis_latest_local.cmd
```

## continuous / finite output

목표 출력은 두 가지입니다.

- continuous recommended voltage waveform
- finite-cycle stop waveform

finite-cycle UI의 primary cycle 선택지는 다음입니다.

- 1.0
- 1.25
- 1.5
- 1.75

`1.75 cycle`은 해당 exact finite-cycle support data가 있을 때 사용합니다. `0.75`는 primary selector에서 제외된 legacy/unexpected cycle이며, `1.75`와 동일시하지 않습니다.

## support/input waveform family

support/input waveform family는 target shape selector가 아닙니다. Sine support를 고르더라도 physical target field shape는 rounded triangle / 100pp fixed입니다.

## 화면에서 확인할 문구

Quick LUT 화면에서는 다음 의미가 보여야 합니다.

- FIELD-ONLY
- rounded triangle
- 100pp fixed
- support/input waveform family
- current / gain / hardware / LCR excluded
- DAQ output fixed: ±5V
- DCAMP Gain fixed: 100%

## 주의

Quick LUT UI가 바뀌지 않은 것처럼 보이면 기존 Streamlit 프로세스를 종료하고 다시 실행하십시오. parent source tree가 repo-local source를 가리는 문제가 없도록 entrypoint guardrail이 들어가 있지만, stale process는 여전히 이전 화면을 보여줄 수 있습니다.
