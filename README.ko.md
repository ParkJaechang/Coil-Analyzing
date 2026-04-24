# Coil Analyzing 한국어 안내

이 저장소는 코일 실험 데이터를 로컬 Streamlit 앱에서 확인하고, Raw Waveforms 검수와 Quick LUT 운용을 수행하기 위한 작업 저장소입니다.

## 빠른 실행

전체 field-analysis 앱:

```powershell
.\launch_field_analysis_latest_local.cmd
```

Quick LUT 중심 운용 앱:

```powershell
.\launch_quick_lut_local.cmd
```

`latest` 앱은 데이터 업로드, Raw Waveforms, diagnostics, validation, export 등 전체 화면을 포함합니다. `quick` 앱은 Quick LUT와 운용에 필요한 화면을 빠르게 여는 용도입니다.

## 한국어 문서

- [사용자 가이드](docs/사용자_가이드.md)
- [데이터 수집 가이드](docs/데이터_수집_가이드.md)
- [Raw Waveforms 데이터 검수 가이드](docs/Raw_Waveforms_데이터_검수_가이드.md)
- [Quick LUT 운용 가이드](docs/Quick_LUT_운용_가이드.md)
- [모델링 정책](docs/모델링_정책.md)
- [용어집](docs/용어집.md)

## 현재 모델링 방향

현재 핵심 목표는 `전압 파형 + 주파수 -> 결과 자기장 파형`입니다. main target field는 rounded triangle / 100pp fixed이며, current/gain/hardware/LCR는 main shape selection 근거가 아닙니다.
