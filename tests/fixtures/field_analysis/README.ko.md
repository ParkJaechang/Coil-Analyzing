# Field Analysis Fixture Dataset

이 디렉터리는 field-analysis parser, Raw Waveforms UI label, finite-cycle policy, quality/anomaly flag 회귀 테스트를 위한 작은 fixture dataset입니다.

이 fixture는 회귀 테스트용이며, 최종 모델 성능 판단용 전체 데이터셋이 아닙니다. 원본 LUT 폴더 전체를 Git에 넣지 않고, 대표 파일만 downsample하여 포함합니다.

## 파일명 규칙

- Continuous: `continuous_{waveform}_{freq}Hz.csv`
- Finite-cycle: `finite_{waveform}_{freq}Hz_{cycle}cycle.csv`

예:

- `continuous_sine_1Hz.csv`
- `continuous_triangle_5Hz.csv`
- `finite_sine_1Hz_1.25cycle.csv`
- `finite_triangle_1Hz_1.75cycle.csv`

## 고정 실험 조건

- DAQ output: `±5V`
- DAQ peak-to-peak: `10Vpp`
- DCAMP Gain: `100%`

## 디렉터리 구성

- `continuous/`: 정상 continuous parser/UI 회귀 fixture
- `finite/`: 정상 finite-cycle parser/UI/policy 회귀 fixture
- `quality_cases/`: 의도적 anomaly를 포함한 quality flag 회귀 fixture
- `expected/`: manifest 기반 expected metadata/quality flags

## Quality Cases

- `finite_nonzero_start_example.csv`: DAQ nonzero-start 의심 케이스
- `finite_spike_example.csv`: magnetic field spike 의심 케이스
- `continuous_current_offset_example.csv`: current offset 의심 케이스
- `finite_truncated_example.csv`: finite active window truncation 의심 케이스

Quality case는 golden usable 데이터가 아니며, `manifest.json`에서 `expected_quality=retest_required`와 `expected_flags`로 구분합니다.

## Git에 넣지 말아야 할 것

- 전체 LUT/raw upload 폴더
- `outputs/field_analysis_app_state` 전체
- runtime cache/export cache
- PR artifacts
- screenshots
- local NAS/export/download cache

## 추후 추가 필요 fixture

현재 요청된 1Hz/5Hz continuous 및 1Hz finite 1.0/1.25/1.5/1.75 sine/triangle 조합은 모두 포함되어 있습니다. 이후 실제 검수에서 suspect로 분류된 추가 케이스가 생기면 `quality_cases/`에 별도 fixture로 추가합니다.
