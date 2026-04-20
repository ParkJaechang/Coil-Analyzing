# Provenance / Quality Badge

## Quality Badge Rule
- metric domain: `bz_effective` (`bz_mT`, global rule `bz_effective = -bz_raw`)
- `재현 양호`: Bz NRMSE <= `0.15`, shape corr >= `0.97`, |phase lag| <= `0.02s`, clipping/saturation 없음
- `주의`: green 기준은 벗어나지만 Bz NRMSE <= `0.30`, shape corr >= `0.90`, |phase lag| <= `0.05s`, clipping/saturation 없음
- `재보정 권장`: clipping/saturation 감지 또는 Bz NRMSE > `0.30` 또는 shape corr < `0.90` 또는 |phase lag| > `0.05s`

## Provenance Fields
- `original_recommendation_id`: corrected lineage의 루트 recommendation ID
- `validation_run_id`: 실제 validation 입력과 연결되는 run key
- `corrected_lut_id`: corrected LUT 산출물 묶음의 canonical ID
- `source_lut_filename`: retune에 사용한 원본 LUT/export 파일명
- `iteration_index`: corrected lineage 내 반복 차수
- `created_at`: validation / corrected 생성 시각
- `correction_rule`: validation residual retune 규칙 문자열
- `before_after_metrics`: target output / bz_effective 도메인의 before/after 비교

## Picker Contract
- `selection_id = <source_kind>::<source_id>`
- `source_kind`는 `recommendation`, `export`, `corrected` 중 하나
- `retune_eligible=true` 인 항목만 standalone picker에서 바로 재사용 가능

## Current Counts
- validation entries: `3`
- corrected LUT entries: `3`
- picker entries: `39`
