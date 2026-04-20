# Operational Scope

- Continuous current exact <= 5 Hz is the only exact path that is operator auto-ready today.
- Continuous field exact <= 5 Hz is software-ready and indexed in the artifacts, but still bench pending.
- Finite exact contains 95 certified measured cells. The one remaining 20 pp sine cell is still provisional preview.
- Anything above 5 Hz is reference only and must not be treated as production exact support.

## Status Rules

| status | meaning |
| --- | --- |
| certified_exact | Measured exact support. Safe for operator use within policy. |
| software_ready_bench_pending | Indexed and software-ready, but still requires bench validation. |
| provisional_experimental | Visible for experiment or preview only. Not certified exact. |
| preview_only | May be useful for analysis, but not for operator exact claims. |
| deprecated | Reference-only or otherwise outside the exact operating path. |
