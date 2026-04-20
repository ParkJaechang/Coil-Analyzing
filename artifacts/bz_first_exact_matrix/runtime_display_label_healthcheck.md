# Runtime Display Label Healthcheck

- success: `True`
- total_records: `311`
- leak_violations: `0`
- display_name_mismatches: `0`
- display_label_mismatches: `0`

| check | status | details |
| --- | --- | --- |
| label_report_present | pass | label_sanitization_report.json loaded |
| display_leak_check | pass | leak_violations=0 |
| display_name_consistency_check | pass | display_name_mismatches=0 |
| display_label_consistency_check | pass | display_label_mismatches=0 |
| label_artifact_coverage_check | pass | {"corrected_lut_catalog": 3, "exact_matrix.continuous_current.cells": 48, "exact_matrix.continuous_current.summary": 10, "exact_matrix.continuous_field.summary": 10, "exact_matrix.finite.cells": 95, "exact_matrix.finite.summary": 48, "exact_matrix.missing": 1, "exact_matrix.provisional": 1, "exact_matrix.reference_only": 7, "lut_catalog": 39, "measurement_roi_priority": 7, "retune_picker_catalog": 39, "validation_catalog": 3} |
