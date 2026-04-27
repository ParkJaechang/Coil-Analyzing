# Raw Waveforms 데이터 검수 가이드

## 목적

Raw Waveforms 화면은 모델링 전에 데이터를 사람이 직접 확인하기 위한 검수 화면입니다. continuous와 finite-cycle 데이터를 모두 확인할 수 있습니다.

## 테스트 선택

`테스트 선택` dropdown은 metadata 기반 label을 표시합니다.

예:

```text
continuous | Sine | 1 Hz | ±5V | Gain 100% | continuous_sine_1Hz.csv
finite-cycle | Triangle | 2 Hz | 1.75 cycle | ±5V | Gain 100% | finite_triangle_2Hz_1.75cycle.csv
```

opaque hash나 internal id는 primary label 앞에 보이지 않아야 합니다.

## Internal ID

`Internal ID`는 앱 내부에서 test를 구분하기 위한 값입니다. 사람이 데이터를 찾을 때는 source file과 metadata label을 우선 사용하고, internal id는 debug/reference 용도로만 사용합니다.

## Source type filter

Source type:

- `all`: continuous와 finite-cycle 모두 표시
- `continuous`: steady-state 또는 continuous 데이터만 표시
- `finite-cycle`: finite-cycle/transient 데이터만 표시

## raw / corrected

- `raw`: 파싱 직후 normalized frame입니다.
- `corrected`: 전처리, sign/offset 등 correction이 반영된 frame입니다.

검수할 때 raw와 corrected를 모두 확인해, 원본 문제인지 전처리 문제인지 구분하십시오.

## finite-cycle marker

finite-cycle 데이터에서는 가능한 경우 다음 marker를 표시합니다.

- detected nonzero start
- detected nonzero end
- target active end
- zero/tail section

marker가 이상하면 start/end detection, truncation, tail 부족, wrong cycle 가능성을 의심해야 합니다.

## anomaly quick checks

Raw Waveforms는 선택한 signal에 대해 다음 값을 표시합니다.

- pp
- rms
- max adjacent jump
- possible spike flag
- possible clipping flag
- flatline suspicion
- duration mismatch suspicion

이 flag는 자동 판정이 아니라 사람 검수 보조 정보입니다.

## Reference test 의미

`비교 기준 테스트 (선택)`은 선택한 파형과 겹쳐 비교할 기준 테스트입니다. 단일 데이터 검수 시에는 `없음`으로 두면 됩니다. 여러 조건을 비교하거나 reference-normalized summary를 보고 싶을 때만 선택합니다.

## 재측정 권장 상황

- spike가 커서 실제 파형보다 순간 jump가 지배적인 경우
- tail이 잘려 predicted_settle_end를 확인할 수 없는 경우
- cycle_count와 duration이 맞지 않는 경우
- raw와 corrected의 부호 또는 scale이 의도와 다른 경우
- source file과 metadata label이 서로 모순되는 경우
