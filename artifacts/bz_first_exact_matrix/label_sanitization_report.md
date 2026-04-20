# Label Sanitization Report

- success: True
- Runtime display fields are checked for leaked hash/internal identifiers and for cross-artifact naming drift.
- display_name is treated as the canonical object label; display_label may add clean context but must stay leak-free and stable within the same source contract.

## Artifact Coverage

| artifact | records |
| --- | --- |
| corrected_lut_catalog | 3 |
| exact_matrix.continuous_current.cells | 48 |
| exact_matrix.continuous_current.summary | 10 |
| exact_matrix.continuous_field.summary | 10 |
| exact_matrix.finite.cells | 95 |
| exact_matrix.finite.summary | 48 |
| exact_matrix.missing | 1 |
| exact_matrix.provisional | 1 |
| exact_matrix.reference_only | 7 |
| lut_catalog | 39 |
| measurement_roi_priority | 7 |
| retune_picker_catalog | 39 |
| validation_catalog | 3 |

## Violations

_None_
