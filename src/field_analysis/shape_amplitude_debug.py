from __future__ import annotations

from typing import Any


def infer_shape_amplitude_debug_fields(
    *,
    request_kind: str,
    target_type: str,
    engine_summary: dict[str, Any],
    confidence_summary: dict[str, Any],
    legacy_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    solver_route = str(engine_summary.get("solver_route") or "")
    plot_source = str(engine_summary.get("plot_source") or "")
    legacy_mode = str((legacy_payload or {}).get("mode") or "")
    template_test_id = (legacy_payload or {}).get("template_test_id")
    support_selection_reason = (
        engine_summary.get("support_selection_reason")
        or confidence_summary.get("support_selection_reason")
    )
    support_family_lock_applied = bool(
        engine_summary.get(
            "support_family_lock_applied",
            confidence_summary.get("support_family_lock_applied", False),
        )
    )

    amplitude_lut_meaning: str = "unknown"
    shape_engine_source: str = "unknown"
    amplitude_engine_source: str = "unknown"
    pp_affects_shape: bool | str = "unknown"

    if request_kind == "size_lut":
        amplitude_lut_meaning = "mixed"
        shape_engine_source = (
            "measured_template_waveform"
            if template_test_id
            else "theoretical_template"
        )
        amplitude_engine_source = "scalar_voltage_lut"
        pp_affects_shape = bool(template_test_id)
    elif solver_route.startswith("harmonic_surface_inverse"):
        shape_engine_source = "harmonic_surface_model"
        amplitude_engine_source = "inverse_voltage_scaling"
        if str(target_type) == "field" and (
            support_family_lock_applied
            or str(support_selection_reason or "") == "exact_family_level_lock"
        ):
            pp_affects_shape = True
    elif solver_route == "finite_exact_direct":
        shape_engine_source = "finite_empirical_support"
        amplitude_engine_source = "finite_support_scaling"
        pp_affects_shape = True
    elif legacy_mode.startswith("harmonic_inverse"):
        shape_engine_source = "legacy_harmonic_support"
        amplitude_engine_source = "inverse_voltage_scaling"
    elif plot_source == "support_blended_preview":
        shape_engine_source = "support_blended_preview"
        amplitude_engine_source = "preview_scale_blend"

    return {
        "amplitude_lut_meaning": amplitude_lut_meaning,
        "shape_engine_source": shape_engine_source,
        "amplitude_engine_source": amplitude_engine_source,
        "pp_affects_shape": pp_affects_shape,
    }
