# Validation / Retune Report

## ??臾몄꽌媛 ?ㅻ챸?섎뒗 ?댁슜

- ?대뼡 異붿쿇 LUT瑜??대뼡 validation run?쇰줈 寃利앺뻽怨? corrected LUT媛 ?대뼸寃??앹꽦?먮뒗吏 ?뺣━?⑸땲??
- 鍮꾧탳 湲곗?? target / predicted / actual / corrected 4媛?怨≪꽑?낅땲??
- ?ы쁽 ?덉쭏 badge, provenance, 二쇱슂 ?꾪뿕 ?좏샇瑜???踰덉뿉 ?뺤씤?????덉뒿?덈떎.

## Provenance
- LUT ID: `real_continuous_current_exact_validation_0p5hz`
- original recommendation id: `continuous_current_exact_real_0p5hz_recommendation`
- validation run id: `6c92e4f555df4fe7b6106580b286b88c_0.5Hz_9V_38gain::2026-04-16T11:35:54`
- corrected LUT id: `real_continuous_current_exact_validation_0p5hz_retuned_control_lut`
- target type: `current`
- waveform: `sine`
- freq_hz: `0.5`
- cycles: `None`
- target level: `39.647` (pp)
- validation test: `6c92e4f555df4fe7b6106580b286b88c_0.5Hz_9V_38gain`
- measured file: `n/a`

## Quality Badge
- label: `재보정 권장`
- tone: `red`
- reasons: `clipping/saturation 감지; NRMSE 70.68%`

## Baseline vs Corrected
- baseline NRMSE: `0.7066`
- corrected NRMSE: `0.7068`
- baseline shape corr: `0.2531`
- corrected shape corr: `1.0000`
- baseline phase lag (s): `-0.287390`
- corrected phase lag (s): `0.000000`
- baseline pp error: `-39.5955`
- corrected pp error: `-39.6298`

## Retune Loop
- iteration_count: `2`
- stop_reason: `iteration_limit_reached`
- within_hardware_limits: `False`
- correction_gain: `0.7`
- improvement_threshold: `0.0`
