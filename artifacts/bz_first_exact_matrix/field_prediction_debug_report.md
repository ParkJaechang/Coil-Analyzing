# Field Prediction Debug Report

- synthetic recommendation cases: `7`
- manual debug cases: `6`
- observed library cases: `30`
- exact field preview contamination in library: `10`

## Manual Root-Cause Cases

- `exact_field_collapse`: source=`exact_field_direct`, status=`unavailable`, reason=`expected_field_near_zero`
- `exact_preview_contamination`: source=`support_blended_preview`, status=`unavailable`, reason=`exact_route_support_blended_preview_bug`
- `zero_fill_fallback`: source=`zero_fill_fallback`, status=`unavailable`, reason=`zero_fill_fallback`
- `same_recipe_surrogate_available`: source=`current_to_bz_surrogate`, status=`available`, reason=`None`
- `validation_transfer`: source=`validation_transfer`, status=`available`, reason=`None`
- `finite_target_leak_suspect`: source=`target_leak_suspect`, status=`available`, reason=`None`

## Synthetic Recommendation Cases

- `continuous_current_exact`: route=`exact`, solver=`harmonic_surface_inverse_exact`, field_source=`current_to_bz_surrogate`, current_source=`exact_current_direct`, auto=`True`
- `continuous_field_exact`: route=`exact`, solver=`harmonic_surface_inverse_exact`, field_source=`exact_field_direct`, current_source=`exact_current_direct`, auto=`True`
- `finite_triangle_field_10`: route=`preview`, solver=`finite_empirical_weighted_support`, field_source=`support_blended_preview`, current_source=`support_blended_preview`, auto=`False`
- `finite_triangle_field_20`: route=`exact`, solver=`finite_exact_direct`, field_source=`exact_field_direct`, current_source=`exact_current_direct`, auto=`True`
- `finite_triangle_field_40`: route=`exact`, solver=`finite_exact_direct`, field_source=`exact_field_direct`, current_source=`exact_current_direct`, auto=`True`
- `finite_triangle_current_20`: route=`exact`, solver=`finite_exact_direct`, field_source=`exact_field_direct`, current_source=`exact_current_direct`, auto=`True`
- `finite_provisional_field_20`: route=`exact`, solver=`finite_exact_direct`, field_source=`exact_field_direct`, current_source=`exact_current_direct`, auto=`True`

