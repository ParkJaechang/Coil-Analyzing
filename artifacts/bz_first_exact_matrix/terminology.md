# Terminology

- The same labels are reused across exact matrix, LUT catalog, and ROI artifacts so operators do not need to translate between documents.
- The status words below are policy words, not generic descriptions.
- If two documents disagree, exact_matrix_final.json and lut_catalog.json take precedence because they are generated from direct scans.

## Glossary

| term | definition |
| --- | --- |
| certified exact | Measured exact support that is counted in the official operating matrix. |
| software-ready bench pending | Operationally indexed in software, but still awaiting validation measurements. |
| provisional preview | A temporary preview route that does not count as certified exact. |
| missing exact | A required exact cell with no measured upload yet. |
| reference only | Visible for analysis or comparison, but not for exact operator operation. |
| validation linked | A LUT entry has at least one tracked validation run in the catalog. |
| corrected LUT | A validation-retuned LUT candidate linked back to its source lineage. |
