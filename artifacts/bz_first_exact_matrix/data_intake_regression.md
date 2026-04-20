# Data Intake Regression

- Data intake is evaluated against direct file scans, filename recovery, and refresh pipeline linkage.
- The bundle is designed so that new files in the memory folders can refresh matrix, catalog, and ROI artifacts together.
- Manifest files remain optional metadata helpers, not the source of truth for inclusion.

## Checks

| check | status | details |
| --- | --- | --- |
| memory_folder_scan | pass | Exact matrix refresh scans uploads/continuous and uploads/transient recursively. |
| manifest_dependency_minimized | pass | Manifest files are optional metadata helpers, not the source of truth for inclusion. |
| folder_alias_typo_supported | pass | Transient sine folder aliases include sinusidal, sinusoidal, sinusoid, sine, and sin. |
| filename_metadata_recovery | pass | Hashed prefixes, p-decimal notation, frequency, cycle count, and level can be reconstructed from file names. |
| refresh_pipeline_links_scope_catalog_roi | pass | refresh_exact_supported_scope.py runs the scope report and then regenerates the bz-first artifact bundle. |
| provisional_promotion_smoke | pass | canonical placeholder keeps the target cell provisional until a non-placeholder 1hz_1cycle_20pp upload arrives, then the cell is promoted to exact. |

## Manifest Visibility

| area | entries |
| --- | --- |
| continuous_entries | 55 |
| transient_entries | 96 |
| validation_entries | 1 |
| lcr_entries | 1 |
