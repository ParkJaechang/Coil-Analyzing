# Matrix Runtime Healthcheck

- success: `True`
- finite_exact_cells: `95`
- provisional_cells: `1`
- missing_exact_cells: `1`
- roi_head_category: `missing_exact_promotion`

| check | status | details |
| --- | --- | --- |
| refresh_exact_supported_scope | pass | refresh_exact_supported_scope.py completed successfully. |
| generate_bz_first_artifacts | pass | generate_bz_first_artifacts.py completed successfully. |
| exact_matrix_count_check | pass | finite_exact_cells=95 |
| provisional_count_check | pass | finite_exact_cells=95, provisional_cells=1, missing_exact_cells=1, promotion_state=provisional_only |
| lut_catalog_count_check | pass | total=39, entries=39, by_status_total=39 |
| catalog_runtime_state_check | pass | groups=29, duplicate_runtime_entries=19, stale_runtime_entries=10 |
| data_intake_regression | pass | passing_checks=6/6 |
| roi_priority_check | pass | head_category=missing_exact_promotion, validation_attention_items=1, validation_present=True |
| promotion_smoke_check | pass | canonical placeholder keeps the target cell provisional until a non-placeholder 1hz_1cycle_20pp upload arrives, then the cell is promoted to exact. |
| runtime_route_consistency_check | pass | 5 passed, 18 deselected in 4.70s |
| runtime_display_label_healthcheck | pass | leak_violations=0, display_name_mismatches=0, display_label_mismatches=0 |
| closeout_summary_check | pass | release_candidate_summary_exists=True |
