# Validation / Retune Report

## 이 문서가 설명하는 내용

- 어떤 추천 LUT를 어떤 validation run으로 검증했고, corrected LUT가 어떻게 생성됐는지 정리합니다.
- 비교 기준은 target / predicted / actual / corrected 4개 곡선입니다.
- 재현 품질 badge, provenance, 주요 위험 신호를 한 번에 확인할 수 있습니다.

## Provenance
- LUT ID: `real_finite_exact_validation_1p25hz_1p25cycle_20pp`
- original recommendation id: `ff132682eb37f728_1.25hz_1.25cycle_20pp`
- validation run id: `8d51f7c793c4e3fa_샘플파형테스트결과_1hz_pp20A::2026-04-16T00:10:58`
- corrected LUT id: `real_finite_exact_validation_1p25hz_1p25cycle_20pp_retuned_control_lut`
- target type: `current`
- waveform: `sine`
- freq_hz: `1.25`
- cycles: `1.25`
- target level: `20.0` (pp)
- validation test: `8d51f7c793c4e3fa_샘플파형테스트결과_1hz_pp20A`
- measured file: `n/a`

## Quality Badge
- label: `재보정 권장`
- tone: `red`
- reasons: `clipping/saturation 감지; NRMSE 70.09%`

## Baseline vs Corrected
- baseline NRMSE: `0.7874`
- corrected NRMSE: `0.7009`
- baseline shape corr: `0.4811`
- corrected shape corr: `nan`
- baseline phase lag (s): `0.135563`
- corrected phase lag (s): `nan`
- baseline pp error: `4.6645`
- corrected pp error: `-19.9970`

## Retune Loop
- iteration_count: `2`
- stop_reason: `improvement_threshold_reached`
- within_hardware_limits: `True`
- correction_gain: `0.7`
- improvement_threshold: `0.0`
