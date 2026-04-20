# Fallback Hierarchy Audit

- hierarchy: `exact_field_direct -> current_to_bz_surrogate -> unavailable`
- unavailable_count: `3`
- surrogate_available_count: `1`
- zero_fill_fallback_count: `1`
- support_blended_preview_bug_count: `1`

- `exact_field_collapse`: source=`exact_field_direct`, status=`unavailable`, reason=`expected_field_near_zero`
- `exact_preview_contamination`: source=`support_blended_preview`, status=`unavailable`, reason=`exact_route_support_blended_preview_bug`
- `zero_fill_fallback`: source=`zero_fill_fallback`, status=`unavailable`, reason=`zero_fill_fallback`
- `same_recipe_surrogate_available`: source=`current_to_bz_surrogate`, status=`available`, reason=`None`
- `validation_transfer`: source=`validation_transfer`, status=`available`, reason=`None`
- `finite_target_leak_suspect`: source=`target_leak_suspect`, status=`available`, reason=`None`
