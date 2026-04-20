# LUT Catalog

- Each LUT is classified against the measured exact matrix, not against stale manifest state.
- Use status plus source_route to decide whether a LUT is operator-safe, bench-pending, preview-only, or deprecated.
- Validation, corrected LUT linkage, and runtime duplicate detection are included so the catalog can serve as an operations handoff document.

## Catalog

| display_label | status | source_route | clipping_risk | validation | corrected | duplicate_runtime | stale_runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current / sine / 0.25 Hz / 20 pp | recommendation | certified_exact | exact_current | low | no | no | no | no |
| current / sine / 0.5 Hz / 20 pp | recommendation | certified_exact | exact_current | low | yes | yes | yes | no |
| current / sine / 0.5 Hz / 20 A | export | certified_exact | exact_current | low | no | no | yes | yes |
| current / sine / 0.5 Hz / 20 A | corrected iter01 | certified_exact | exact_current | high | yes | yes | yes | yes |
| current / sine / 0.5 Hz / 1 cycle / 20 pp | export | certified_exact | finite_exact | low | no | no | no | no |
| current / triangle / 0.5 Hz / 1 cycle / 20 pp | export | certified_exact | finite_exact | low | no | no | no | no |
| current / triangle / 1 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | no | no |
| current / sine / 1 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | yes | yes |
| current / triangle / 1 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | yes | yes |
| current / sine / 1 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | yes | no |
| current / triangle / 1 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | yes | no |
| current / triangle / 1.25 Hz / 1.25 cycle / 10 pp | export | certified_exact | finite_exact | low | no | no | no | no |
| current / sine / 1.25 Hz / 1.25 cycle / 20 pp | corrected iter01 | certified_exact | finite_exact | low | yes | yes | no | no |
| current / triangle / 3 Hz / 1.5 cycle / 20 pp | export | certified_exact | finite_exact | low | no | no | no | no |
| field / triangle / 0.5 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | yes | yes |
| field / triangle / 0.5 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | yes | no |
| field / triangle / 0.5 Hz / 1.25 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | no | no |
| field / triangle / 1.25 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | yes | yes |
| field / triangle / 1.25 Hz / 1 cycle / 20 pp | recommendation | certified_exact | finite_exact | low | no | no | yes | no |
| field / sine / 0.25 Hz / 20 pp | recommendation | software_ready_bench_pending | exact_field | high | yes | yes | yes | no |
| field / triangle / 0.25 Hz / 20 pp | recommendation | software_ready_bench_pending | exact_field | high | no | no | no | no |
| field / sine / 0.25 Hz / 20 A | corrected iter01 | software_ready_bench_pending | exact_field | low | yes | yes | yes | yes |
| field / sine / 0.25 Hz / 50 pp | recommendation | software_ready_bench_pending | exact_field | low | no | no | no | no |
| field / triangle / 0.25 Hz / 50 pp | recommendation | software_ready_bench_pending | exact_field | low | no | no | no | no |
| field / sine / 0.25 Hz / 116.226 A | export | software_ready_bench_pending | exact_field | low | no | no | no | no |
| field / triangle / 1 Hz / 50 pp | recommendation | software_ready_bench_pending | exact_field | low | no | no | no | no |
| current / sine / 1 Hz / 1 cycle / 20 pp | recommendation | provisional_experimental | provisional_preview | low | no | no | no | no |
| current / sine / 0.75 Hz / 20 pp | recommendation | preview_only | preview_only | low | no | no | no | no |
| field / triangle / 1 Hz / 1.25 cycle / 100 pp | recommendation | preview_only | preview_only | low | no | no | yes | yes |
| field / triangle / 1 Hz / 1.25 cycle / 100 pp | recommendation | preview_only | preview_only | low | no | no | yes | no |
| field / sine / 1.25 Hz / 19.999 A | recommendation | preview_only | preview_only | high | no | no | no | no |
| field / triangle / 1.25 Hz / 50 pp | recommendation | preview_only | preview_only | low | no | no | no | no |
| field / triangle / 1.25 Hz / 100 pp | recommendation | preview_only | preview_only | low | no | no | no | no |
| field / sine / 1.25 Hz / 1.25 cycle / 19.9996 A | recommendation | preview_only | preview_only | low | no | no | no | no |
| field / sine / 1.25 Hz / 1.25 cycle / 40 pp | recommendation | preview_only | preview_only | low | no | no | no | no |
| field / triangle / 1.25 Hz / 1.25 cycle / 40 pp | recommendation | preview_only | preview_only | low | no | no | yes | yes |
| field / triangle / 1.25 Hz / 1.25 cycle / 40 pp | recommendation | preview_only | preview_only | low | no | no | yes | no |
| current / triangle / 6 Hz / 1.25 cycle / 20 pp | recommendation | deprecated | reference_only | low | no | no | yes | yes |
| current / triangle / 6 Hz / 1.25 cycle / 20 pp | recommendation | deprecated | reference_only | low | no | no | yes | no |
