# Known Limitations

- Continuous field exact remains bench pending even though the software can index it.
- Finite exact is still 95 certified cells plus one provisional preview cell, not a full 96 certified cells yet.
- Anything above 5 Hz is reference-only and should not be interpreted as exact support.
- Filename inference is robust but not magical; severely malformed file names can still block automatic classification.

## Current Limits

- `app_ui.py` is intentionally out of scope for this thread.
- Interpolated auto remains closed by policy.
- Validation-corrected LUTs improve lineage visibility, but they do not change exact policy on their own.
