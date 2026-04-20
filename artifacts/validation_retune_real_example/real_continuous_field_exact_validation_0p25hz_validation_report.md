# Validation / Retune Report

## ??臾몄꽌媛 ?ㅻ챸?섎뒗 ?댁슜

- ?대뼡 異붿쿇 LUT瑜??대뼡 validation run?쇰줈 寃利앺뻽怨? corrected LUT媛 ?대뼸寃??앹꽦?먮뒗吏 ?뺣━?⑸땲??
- 鍮꾧탳 湲곗?? target / predicted / actual / corrected 4媛?怨≪꽑?낅땲??
- ?ы쁽 ?덉쭏 badge, provenance, 二쇱슂 ?꾪뿕 ?좏샇瑜???踰덉뿉 ?뺤씤?????덉뒿?덈떎.

## Provenance
- LUT ID: `real_continuous_field_exact_validation_0p25hz`
- original recommendation id: `continuous_field_exact_real_0p25hz_recommendation`
- validation run id: `a45d8cb65a6b4e36b21481394aec8c0b_0.25Hz_9V_36gain::2026-04-16T11:35:55`
- corrected LUT id: `real_continuous_field_exact_validation_0p25hz_retuned_control_lut`
- target type: `field`
- waveform: `sine`
- freq_hz: `0.25`
- cycles: `None`
- target level: `276.50419999999997` (pp)
- validation test: `a45d8cb65a6b4e36b21481394aec8c0b_0.25Hz_9V_36gain`
- measured file: `n/a`

## Quality Badge
- label: `재보정 권장`
- tone: `red`
- reasons: `clipping/saturation 감지; NRMSE 66.67%`

## Baseline vs Corrected
- baseline NRMSE: `0.6907`
- corrected NRMSE: `0.6667`
- baseline shape corr: `0.3125`
- corrected shape corr: `0.9979`
- baseline phase lag (s): `0.754643`
- corrected phase lag (s): `0.000000`
- baseline pp error: `-248.8941`
- corrected pp error: `-259.2703`

## Retune Loop
- iteration_count: `2`
- stop_reason: `iteration_limit_reached`
- within_hardware_limits: `False`
- correction_gain: `0.7`
- improvement_threshold: `0.0`
