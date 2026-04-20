# Validation Data Template

- Validation uploads should make it obvious which LUT lineage and operating point they belong to.
- Use explicit metadata whenever possible because validation artifacts are read later by audit scripts.
- Attach the measured file, the source LUT id, and the intended scenario folder together.

## Template

- Scenario folder: `uploads/validation/<continuous_current_exact | continuous_field_exact | finite_sine_exact | finite_triangle_exact>`
- Source LUT id: `<recommendation id or corrected LUT id>`
- Original recommendation id: `<lineage root id>`
- Validation file name: `<bench hash>_<scenario>_<freq token>_<level token>.csv`
- Expected outputs after refresh: validation_catalog, corrected_lut_catalog, retune_picker_catalog
