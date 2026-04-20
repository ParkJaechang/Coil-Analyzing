# Quick Start Summary

- For operators, the minimum reading order is operator_workflow -> operational_scope -> exact_matrix_final -> measurement_roi_priority.
- For intake/debug work, add data_contract -> filename_convention -> data_intake_regression.
- For LUT validation work, add lut_catalog -> lut_audit_report -> validation_retune_overview.

## Bundle Contents

| artifact | role |
| --- | --- |
| exact_matrix_final | exact/provisional/missing/reference-only source of truth |
| lut_catalog | per-LUT operational classification and lineage linkage |
| measurement_roi_priority | next best measurement and validation actions |
| data_intake_regression | evidence that direct scans and refresh linkage still work |
