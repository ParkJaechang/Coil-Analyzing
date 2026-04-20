# LUT Audit Report

- This report highlights what is ready for operators and what still needs validation, promotion, or retirement.
- Deprecated means reference-only, above-band, or otherwise outside the certified exact operating path.
- A provisional LUT can exist in the catalog without upgrading the exact matrix until a measured exact upload arrives, and duplicate runtime rows are flagged explicitly.

## Status Summary

| status | count |
| --- | --- |
| certified_exact | 19 |
| software_ready_bench_pending | 7 |
| provisional_experimental | 1 |
| preview_only | 10 |
| deprecated | 2 |

## Needs Attention

| display_label | status | source_route | validation | corrected | clipping_risk | duplicate_runtime | stale_runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current / sine / 0.25 Hz / 20 pp | recommendation | certified_exact | exact_current | no | no | low | no | no |
| current / sine / 0.5 Hz / 20 pp | recommendation | certified_exact | exact_current | yes | yes | low | yes | no |
| current / sine / 0.5 Hz / 20 A | export | certified_exact | exact_current | no | no | low | yes | yes |
| current / sine / 0.5 Hz / 20 A | corrected iter01 | certified_exact | exact_current | yes | yes | high | yes | yes |
| current / sine / 0.5 Hz / 1 cycle / 20 pp | export | certified_exact | finite_exact | no | no | low | no | no |
| current / triangle / 0.5 Hz / 1 cycle / 20 pp | export | certified_exact | finite_exact | no | no | low | no | no |
| current / triangle / 1 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | no | no |
| current / sine / 1 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | yes | yes |
| current / triangle / 1 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | yes | yes |
| current / sine / 1 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | yes | no |
| current / triangle / 1 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | yes | no |
| current / triangle / 1.25 Hz / 1.25 cycle / 10 pp | export | certified_exact | finite_exact | no | no | low | no | no |
| current / triangle / 3 Hz / 1.5 cycle / 20 pp | export | certified_exact | finite_exact | no | no | low | no | no |
| field / triangle / 0.5 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | yes | yes |
| field / triangle / 0.5 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | yes | no |
| field / triangle / 0.5 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | no | no |
| field / triangle / 1.25 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | yes | yes |
| field / triangle / 1.25 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | no | no | low | yes | no |
| field / sine / 0.25 Hz / 20 pp | recommendation | software_ready_bench_pending | exact_field | yes | yes | high | yes | no |
| field / triangle / 0.25 Hz / 20 pp | recommendation | software_ready_bench_pending | exact_field | no | no | high | no | no |
| field / sine / 0.25 Hz / 20 A | corrected iter01 | software_ready_bench_pending | exact_field | yes | yes | low | yes | yes |
| field / sine / 0.25 Hz / 50 pp | recommendation | software_ready_bench_pending | exact_field | no | no | low | no | no |
| field / triangle / 0.25 Hz / 50 pp | recommendation | software_ready_bench_pending | exact_field | no | no | low | no | no |
| field / sine / 0.25 Hz / 116.226 A | export | software_ready_bench_pending | exact_field | no | no | low | no | no |
| field / triangle / 1 Hz / 50 pp | recommendation | software_ready_bench_pending | exact_field | no | no | low | no | no |
| current / sine / 1 Hz / 1 cycle / 20 pp | recommendation | provisional_experimental | provisional_preview | no | no | low | no | no |
| current / sine / 0.75 Hz / 20 pp | recommendation | preview_only | preview_only | no | no | low | no | no |
| field / triangle / 1 Hz / 1.25 cycle / 100 pp | recommendation | preview_only | preview_only | no | no | low | yes | yes |
| field / triangle / 1 Hz / 1.25 cycle / 100 pp | recommendation | preview_only | preview_only | no | no | low | yes | no |
| field / sine / 1.25 Hz / 19.999 A | recommendation | preview_only | preview_only | no | no | high | no | no |
| field / triangle / 1.25 Hz / 50 pp | recommendation | preview_only | preview_only | no | no | low | no | no |
| field / triangle / 1.25 Hz / 100 pp | recommendation | preview_only | preview_only | no | no | low | no | no |
| field / sine / 1.25 Hz / 1.25 cycle / 19.9996 A | recommendation | preview_only | preview_only | no | no | low | no | no |
| field / sine / 1.25 Hz / 1.25 cycle / 40 pp | recommendation | preview_only | preview_only | no | no | low | no | no |
| field / triangle / 1.25 Hz / 1.25 cycle / 40 pp | recommendation | preview_only | preview_only | no | no | low | yes | yes |
| field / triangle / 1.25 Hz / 1.25 cycle / 40 pp | recommendation | preview_only | preview_only | no | no | low | yes | no |
| current / triangle / 6 Hz / 1.25 cycle / 20 pp | recommendation | deprecated | reference_only | no | no | low | yes | yes |
| current / triangle / 6 Hz / 1.25 cycle / 20 pp | recommendation | deprecated | reference_only | no | no | low | yes | no |
