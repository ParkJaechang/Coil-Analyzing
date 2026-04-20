# Validation Retune Overview

- Validation and retune use the same backend for continuous current exact, continuous field exact, and finite exact.
- Validation results do not automatically upgrade a preview or provisional cell to certified exact. Only measured exact uploads do that.
- Quality is evaluated in the Bz domain and follows the global rule `bz_effective = -bz_raw`.
- Use the LUT catalog to see which LUTs already have validation lineage and corrected LUT artifacts.

## Quality Rule

## Quality Badge Rule
- metric domain: `bz_effective` (`bz_mT`, global rule `bz_effective = -bz_raw`)
- `재현 양호`: Bz NRMSE <= `0.15`, shape corr >= `0.97`, |phase lag| <= `0.02s`, clipping/saturation 없음
- `주의`: green 기준은 벗어나지만 Bz NRMSE <= `0.30`, shape corr >= `0.90`, |phase lag| <= `0.05s`, clipping/saturation 없음
- `재보정 권장`: clipping/saturation 감지 또는 Bz NRMSE > `0.30` 또는 shape corr < `0.90` 또는 |phase lag| > `0.05s`
