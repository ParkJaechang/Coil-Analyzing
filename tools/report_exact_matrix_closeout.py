from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ARTIFACT_DIR = Path("artifacts/policy_eval")
SCOPE_JSON = ARTIFACT_DIR / "exact_and_finite_scope.json"
OPERATIONS_JSON = ARTIFACT_DIR / "operational_support_scope.json"
EXPORT_JSON = ARTIFACT_DIR / "exact_export_validation.json"
FINITE_JSON = ARTIFACT_DIR / "finite_exact_productization.json"
UI_JSON = ARTIFACT_DIR / "exact_matrix_ui_validation.json"
ROUTER_JSON = ARTIFACT_DIR / "request_router_validation.json"
BROWSER_JSON = ARTIFACT_DIR / "browser_export_validation.json"
OUTPUT_JSON = ARTIFACT_DIR / "exact_matrix_closeout.json"
OUTPUT_MD = ARTIFACT_DIR / "exact_matrix_closeout.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    scope = _load_json(SCOPE_JSON)
    operations = _load_json(OPERATIONS_JSON)
    export_validation = _load_json(EXPORT_JSON)
    finite_productization = _load_json(FINITE_JSON)
    ui_validation = _load_json(UI_JSON)
    router_validation = _load_json(ROUTER_JSON)
    browser_validation = _load_json(BROWSER_JSON)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_support_band_hz": scope["official_support_band_hz"],
        "continuous_support": {
            "current_exact_auto": operations["internal_operations"]["current_exact_auto"],
            "field_exact_status": operations["internal_operations"]["field_exact"],
            "interpolated_auto_enabled": operations["internal_operations"]["interpolation"],
            "continuous_exact_grid_candidates": scope["continuous_exact_grid_candidates"]["recommended_next_measurements"],
        },
        "finite_support": {
            "official_recipe_total": scope["finite_all_exact_scope"]["official_recipe_total"],
            "sine_exact_total": scope["finite_exact_scope"]["total_files"],
            "triangle_exact_total": scope["finite_triangle_exact_scope"]["total_files"],
            "remaining_exact_gaps": scope["finite_all_exact_scope"]["missing_exact_combinations"],
            "provisional_preview_combinations": scope["finite_all_exact_scope"].get("provisional_preview_combinations", []),
            "nearest_recipe_guidance_enabled": True,
        },
        "ui_state_validation": ui_validation["cases"],
        "request_router_validation": router_validation["cases"],
        "export_validation_cases": export_validation["cases"],
        "triangle_exact_productization": finite_productization,
        "browser_export_validation": {
            "scope": browser_validation.get("scope"),
            "download_content_validation_passed": browser_validation.get("download_content_validation_passed"),
            "clicked_download_buttons": browser_validation.get("clicked_download_buttons", []),
            "downloaded_files": browser_validation.get("downloaded_files", []),
            "limitations": browser_validation.get("limitations", []),
        },
        "recommended_next_measurements": operations["user_facing"].get("recommended_next_measurements", []),
    }

    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Exact Matrix Closeout",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- official_support_band_hz: `{payload['official_support_band_hz']['min']} ~ {payload['official_support_band_hz']['max']}`",
        "",
        "## Continuous",
        "",
        f"- current_exact_auto: `{payload['continuous_support']['current_exact_auto']}`",
        f"- field_exact_status: `{payload['continuous_support']['field_exact_status']}`",
        f"- interpolated_auto_enabled: `{payload['continuous_support']['interpolated_auto_enabled']}`",
        f"- continuous_exact_grid_candidates: `{payload['continuous_support']['continuous_exact_grid_candidates']}`",
        "",
        "## Finite",
        "",
        f"- official_recipe_total: `{payload['finite_support']['official_recipe_total']}`",
        f"- sine_exact_total: `{payload['finite_support']['sine_exact_total']}`",
        f"- triangle_exact_total: `{payload['finite_support']['triangle_exact_total']}`",
        f"- remaining_exact_gaps: `{payload['finite_support']['remaining_exact_gaps']}`",
        f"- provisional_preview_combinations: `{payload['finite_support']['provisional_preview_combinations']}`",
        "",
            "## UI State Validation",
            "",
            "| case | support_state | preview_only | allow_auto_download | policy_version |",
            "| --- | --- | --- | --- | --- |",
        ]
    for case in ui_validation["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["name"]),
                    str(case["support_state"]),
                    str(case["preview_only"]),
                    str(case["allow_auto_download"]),
                    str(case["policy_version"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Request Router Validation",
            "",
            "| case | clicked | freq_before | freq_after | target_after | finite_mode_after | cycle_after |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in router_validation["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case["name"]),
                    str(case.get("clicked_button_label", "")),
                    str(case.get("freq_hz_before", "")),
                    str(case.get("freq_hz_after", "")),
                    str(case.get("target_after", "")),
                    str(case.get("finite_mode_after", "")),
                    str(case.get("cycle_after", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Recommended Next Measurements",
            "",
            *[f"- `{item}`" for item in payload["recommended_next_measurements"]],
            "",
            "## Browser Export Validation",
            "",
            f"- scope: `{payload['browser_export_validation']['scope']}`",
            f"- download_content_validation_passed: `{payload['browser_export_validation']['download_content_validation_passed']}`",
            f"- clicked_download_buttons: `{payload['browser_export_validation']['clicked_download_buttons']}`",
            f"- downloaded_files: `{[item['name'] for item in payload['browser_export_validation']['downloaded_files']]}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
