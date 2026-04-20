# Support Route Level Influence

- group_count: `3`
- shape_affected_group_count: `3`
- support_switch_group_count: `3`
- source_switch_group_count: `1`

## continuous_current_exact_sine_0p5hz
- levels: `[5.0, 15.0, 30.0]`
- pp_affects_shape: `True`
- reason_codes: `['support_id_switch']`
- selected_support_ids: `['4816a91ef9d87e20_sine_0.5_20__sine__0.5Hz__20App', '7b60346ff50fb380_sine_0.5_30__sine__0.5Hz__30App', '8e8715b5b45002bb_sine_0.5_5__sine__0.5Hz__5App']`
- prediction_sources: `['current_to_bz_surrogate:available']`

## continuous_field_exact_sine_0p25hz
- levels: `[20.0, 91.21721762947975, 192.09783560360785, 0.0]`
- pp_affects_shape: `True`
- reason_codes: `['support_id_switch', 'prediction_source_switch', 'limit_induced_switch']`
- selected_support_ids: `['5263f7a6ac112f7a_sine_0.25_20__sine__0.25Hz__20App', 'aff1e72e2b9707c7_sine_0.25_30__sine__0.25Hz__30App', 'bdf6e594a99ce4f0_sine_0.25_10__sine__0.25Hz__10App', 'd0c3732104a94580_sine_0.25_5__sine__0.25Hz__5App']`
- prediction_sources: `['current_to_bz_surrogate:available', 'exact_field_direct:available', 'support_blended_preview:unavailable']`

## finite_exact_triangle_1p25hz_1p25cycle
- levels: `[10.0, 20.0]`
- pp_affects_shape: `True`
- reason_codes: `['support_id_switch']`
- selected_support_ids: `['1.25hz_1.25cycle_10pp__triangle__1.25Hz__10App', '1.25hz_1.25cycle_20pp__triangle__1.25Hz__20App']`
- prediction_sources: `['exact_field_direct:available']`
