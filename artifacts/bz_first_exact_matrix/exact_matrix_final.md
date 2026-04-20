# Exact Matrix Final

- This file is the operator-facing source of truth for certified exact, provisional, missing, and reference-only cells.
- Continuous current exact <= 5 Hz is the only auto-operational exact path.
- Continuous field exact <= 5 Hz is software-ready but still bench pending.
- Finite exact currently has 95 certified cells, 1 provisional cell(s), and 1 missing exact cell(s).

## Continuous Current Exact Matrix

| display_label | status |
| --- | --- |
| current / sine / 0.25 Hz | levels 5 A, 10 A, 15 A, 20 A, 30 A | certified_exact |
| current / sine / 0.5 Hz | levels 5 A, 10 A, 15 A, 20 A, 30 A | certified_exact |
| current / sine / 1 Hz | levels 5 A, 10 A, 15 A, 20 A, 30 A | certified_exact |
| current / sine / 2 Hz | levels 5 A, 10 A, 15 A, 20 A, 30 A | certified_exact |
| current / sine / 5 Hz | levels 5 A, 10 A, 15 A, 20 A | certified_exact |
| current / triangle / 0.25 Hz | levels 5 A, 10 A, 15 A, 20 A, 30 A | certified_exact |
| current / triangle / 0.5 Hz | levels 5 A, 10 A, 15 A, 20 A, 30 A | certified_exact |
| current / triangle / 1 Hz | levels 5 A, 10 A, 15 A, 20 A, 30 A | certified_exact |
| current / triangle / 2 Hz | levels 5 A, 10 A, 15 A, 20 A, 30 A | certified_exact |
| current / triangle / 5 Hz | levels 5 A, 10 A, 15 A, 20 A | certified_exact |

## Continuous Field Exact Matrix

| display_label | status | bench_validation |
| --- | --- | --- |
| field / sine / 0.25 Hz | software_ready_bench_pending | pending |
| field / sine / 0.5 Hz | software_ready_bench_pending | pending |
| field / sine / 1 Hz | software_ready_bench_pending | pending |
| field / sine / 2 Hz | software_ready_bench_pending | pending |
| field / sine / 5 Hz | software_ready_bench_pending | pending |
| field / triangle / 0.25 Hz | software_ready_bench_pending | pending |
| field / triangle / 0.5 Hz | software_ready_bench_pending | pending |
| field / triangle / 1 Hz | software_ready_bench_pending | pending |
| field / triangle / 2 Hz | software_ready_bench_pending | pending |
| field / triangle / 5 Hz | software_ready_bench_pending | pending |

## Finite Exact Matrix

| display_label | status |
| --- | --- |
| current / sine / 0.25 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 0.25 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 0.25 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 0.25 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 0.5 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 0.5 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 0.5 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 0.5 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 1 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 1 Hz / 1 cycle | levels 10 pp | certified_exact |
| current / sine / 1 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 1 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 1.25 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 1.25 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 1.25 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 1.25 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 2 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 2 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 2 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 2 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 5 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 5 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 5 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / sine / 5 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 0.25 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 0.25 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 0.25 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 0.25 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 0.5 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 0.5 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 0.5 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 0.5 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 1 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 1 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 1 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 1 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 1.25 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 1.25 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 1.25 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 1.25 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 2 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 2 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 2 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 2 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 3 Hz / 0.75 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 3 Hz / 1 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 3 Hz / 1.25 cycle | levels 10 pp, 20 pp | certified_exact |
| current / triangle / 3 Hz / 1.5 cycle | levels 10 pp, 20 pp | certified_exact |

## Provisional Cell

| display_label | source_exact_level_pp_a | scale_ratio | status |
| --- | --- | --- | --- |
| current / sine / 1 Hz / 1 cycle / 20 pp | 10 | 2 | provisional_preview |

## Missing Exact Cell

| display_label | status | promotion_target |
| --- | --- | --- |
| current / sine / 1 Hz / 1 cycle / 20 pp | missing_exact | 96 exact recipes after measured upload arrives. |

## Reference Only

| display_label | status |
| --- | --- |
| current / sine / 10 Hz / 5 A | reference_only |
| current / sine / 10 Hz / 10 A | reference_only |
| current / sine / 15 Hz / 5 A | reference_only |
| current / sine / 15 Hz / 10 A | reference_only |
| current / triangle / 10 Hz / 5 A | reference_only |
| current / triangle / 10 Hz / 10 A | reference_only |
| current / triangle / 15 Hz / 5 A | reference_only |
