# Operational Support Scope

- generated_at_utc: `2026-04-17T07:11:53.751481+00:00`
- official_support_band_hz: `0.25 ~ 5`

## User-Facing Scope

### Auto Available
- continuous / current / exact support only, <= 5 Hz
- finite / exact-supported recipe only, <= 5 Hz

### Exact, Bench Pending
- continuous / field / exact support only, <= 5 Hz (software-ready, bench pending)

### Finite Exact Scope
- finite exact recipe table: sine 47 + triangle 48, <= 5 Hz
- provisional preview fallback remains enabled for unsupported finite cells with an approved substitute recipe

### Preview Only
- interpolated_in_hull requests
- finite requests outside the exact recipe table
- continuous exact requests above 5 Hz

### Blocked
- interpolated_edge
- out_of_hull
- field target auto

### Reference Only
- continuous support above 5 Hz remains reference-only

- fallback_rule: `Unsupported requests stay in preview-only or blocked state and should show the nearest exact recipe within the <= 5 Hz official band when available.`

### Recommended Next Measurements
- continuous sine exact grid: 0.75 Hz @ 5/10/20 A
- continuous sine exact grid: 1.5 Hz @ 5/10/20 A
- continuous sine exact grid: 3.0 Hz @ 5/10/20 A
- continuous sine exact grid: 4.0 Hz @ 5/10/20 A

## Internal Operations

| path | status | scope | note |
| --- | --- | --- | --- |
| continuous/current exact | operational | <= 5 Hz exact support only | policy v2 |
| continuous/field exact | software_ready_bench_pending | <= 5 Hz exact support only | bench sign-off pending |
| finite exact | software_ready_exact_recipe_only | <= 5 Hz exact-supported recipes only | sine 47 + triangle 48 |
| interpolation auto | closed | preview-only | disabled pending future R&D |
| finite generalization | preview_only | research-only | preview quality not operational |
| continuous > 5 Hz | reference_only | excluded from official operation | experimental/reference only |