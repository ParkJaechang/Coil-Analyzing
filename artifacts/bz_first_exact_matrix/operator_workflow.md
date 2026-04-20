# Operator Workflow

- Read exact_matrix_final first. It is the source of truth for what is certified exact, bench pending, provisional, missing, or reference only.
- Use only certified exact rows for auto operation. Treat software-ready bench pending as validation backlog, not as production-ready.
- After adding new measurement files to uploads, run refresh_exact_supported_scope.py to rebuild scope, exact matrix, LUT catalog, and ROI artifacts.
- If the missing sine / 1.0 Hz / 1.0 cycle / 20 pp file is uploaded, check that the provisional cell disappears and the finite exact count increases by one.

## Daily Flow

1. Check exact_matrix_final.md.
2. Check lut_catalog.md if a specific LUT must be traced.
3. Add new files using filename_convention.md.
4. Run `python tools/refresh_exact_supported_scope.py`.
5. Re-open exact_matrix_final.md, measurement_roi_priority.md, and data_intake_regression.md.
