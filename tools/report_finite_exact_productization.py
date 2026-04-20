from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


SCOPE_JSON = Path("artifacts/policy_eval/exact_and_finite_scope.json")
EXPORT_JSON = Path("artifacts/policy_eval/exact_export_validation.json")
UI_JSON = Path("artifacts/policy_eval/exact_matrix_ui_validation.json")
OUTPUT_JSON = Path("artifacts/policy_eval/finite_exact_productization.json")
OUTPUT_MD = Path("artifacts/policy_eval/finite_exact_productization.md")

TRIANGLE_CASES = [
    "finite_exact_triangle_0p5hz_1p0cycle_20pp",
    "finite_exact_triangle_1p25hz_1p25cycle_10pp",
    "finite_exact_triangle_3hz_1p5cycle_20pp",
]


def _ui_case_or_placeholder(ui_cases: dict[str, dict], *names: str) -> dict:
    for name in names:
        case = ui_cases.get(name)
        if isinstance(case, dict):
            return case
    return {
        "name": names[0] if names else "not_evaluated",
        "support_state": "not_evaluated",
        "preview_only": False,
        "allow_auto_download": False,
        "warning_messages": [],
        "info_messages": [],
        "success_messages": [],
        "caption_messages": [],
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_case(case: dict) -> dict:
    return {
        "support_state": case["support_state"],
        "preview_only": case["preview_only"],
        "allow_auto_download": case["allow_auto_download"],
        "profile_freq_hz": case["profile_freq_hz"],
        "target_waveform": case["target_waveform"],
        "target_type": case["target_type"],
        "lut_row_count": case["lut_row_count"],
        "limited_voltage_pp_v": case["limited_voltage_pp_v"],
        "time_end_s": case["time_end_s"],
        "csv_file": case["csv_file"],
        "lut_file": case["lut_file"],
        "formula_file": case["formula_file"],
    }


def _trim_messages(values: list[str]) -> list[str]:
    return [value for value in values if value][:5]


def main() -> int:
    scope = _load_json(SCOPE_JSON)
    export_validation = _load_json(EXPORT_JSON)["cases"]
    ui_cases = {case["name"]: case for case in _load_json(UI_JSON)["cases"]}

    triangle_ui = _ui_case_or_placeholder(
        ui_cases,
        "finite_exact_triangle_auto",
        "finite_provisional_sine_preview",
    )
    provisional_ui = _ui_case_or_placeholder(ui_cases, "finite_provisional_sine_preview")
    missing_ui = _ui_case_or_placeholder(ui_cases, "finite_unsupported_triangle")
    triangle_export_cases = {
        name: _compact_case(export_validation[name])
        for name in TRIANGLE_CASES
    }

    finite_all = scope["finite_all_exact_scope"]
    sine_scope = scope["finite_exact_scope"]
    triangle_scope = scope["finite_triangle_exact_scope"]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_support_band_hz": scope["official_support_band_hz"],
        "finite_exact_totals": {
            "official_recipe_total": finite_all["official_recipe_total"],
            "sine_total": sine_scope["total_files"],
            "triangle_total": triangle_scope["total_files"],
        },
        "remaining_exact_gaps": finite_all["missing_exact_combinations"],
        "provisional_preview_combinations": finite_all.get("provisional_preview_combinations", []),
        "triangle_representative_export_validation": triangle_export_cases,
        "triangle_exact_ui_state": {
            "support_state": triangle_ui["support_state"],
            "preview_only": triangle_ui["preview_only"],
            "allow_auto_download": triangle_ui["allow_auto_download"],
            "warnings": _trim_messages(triangle_ui["warning_messages"]),
            "infos": _trim_messages(triangle_ui["info_messages"]),
        },
        "triangle_missing_recipe_ui_state": {
            "support_state": missing_ui["support_state"],
            "preview_only": missing_ui["preview_only"],
            "allow_auto_download": missing_ui["allow_auto_download"],
            "warnings": _trim_messages(missing_ui["warning_messages"]),
            "infos": _trim_messages(missing_ui["info_messages"]),
        },
        "provisional_gap_ui_state": {
            "support_state": provisional_ui["support_state"],
            "preview_only": provisional_ui["preview_only"],
            "allow_auto_download": provisional_ui["allow_auto_download"],
            "warnings": _trim_messages(provisional_ui["warning_messages"]),
            "infos": _trim_messages(provisional_ui["info_messages"]),
        },
    }

    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Finite Exact Productization",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- official_support_band_hz: `{payload['official_support_band_hz']['min']} ~ {payload['official_support_band_hz']['max']}`",
        f"- official_recipe_total: `{payload['finite_exact_totals']['official_recipe_total']}`",
        f"- sine_total: `{payload['finite_exact_totals']['sine_total']}`",
        f"- triangle_total: `{payload['finite_exact_totals']['triangle_total']}`",
        f"- remaining_exact_gaps: `{payload['remaining_exact_gaps']}`",
        f"- provisional_preview_combinations: `{payload['provisional_preview_combinations']}`",
        "",
        "## Triangle Representative Export Validation",
        "",
    ]
    for name, case in triangle_export_cases.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- support_state: `{case['support_state']}`",
                f"- preview_only: `{case['preview_only']}`",
                f"- allow_auto_download: `{case['allow_auto_download']}`",
                f"- profile_freq_hz: `{case['profile_freq_hz']}`",
                f"- limited_voltage_pp_v: `{case['limited_voltage_pp_v']}`",
                f"- time_end_s: `{case['time_end_s']}`",
                f"- csv_file: `{case['csv_file']}`",
                f"- lut_file: `{case['lut_file']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Triangle Exact UI State",
            "",
            f"- support_state: `{payload['triangle_exact_ui_state']['support_state']}`",
            f"- preview_only: `{payload['triangle_exact_ui_state']['preview_only']}`",
            f"- allow_auto_download: `{payload['triangle_exact_ui_state']['allow_auto_download']}`",
            f"- warnings: `{payload['triangle_exact_ui_state']['warnings']}`",
            f"- infos: `{payload['triangle_exact_ui_state']['infos']}`",
            "",
            "## Missing Triangle Recipe UI State",
            "",
            f"- support_state: `{payload['triangle_missing_recipe_ui_state']['support_state']}`",
            f"- preview_only: `{payload['triangle_missing_recipe_ui_state']['preview_only']}`",
            f"- allow_auto_download: `{payload['triangle_missing_recipe_ui_state']['allow_auto_download']}`",
            f"- warnings: `{payload['triangle_missing_recipe_ui_state']['warnings']}`",
            f"- infos: `{payload['triangle_missing_recipe_ui_state']['infos']}`",
            "",
            "## Provisional Gap UI State",
            "",
            f"- support_state: `{payload['provisional_gap_ui_state']['support_state']}`",
            f"- preview_only: `{payload['provisional_gap_ui_state']['preview_only']}`",
            f"- allow_auto_download: `{payload['provisional_gap_ui_state']['allow_auto_download']}`",
            f"- warnings: `{payload['provisional_gap_ui_state']['warnings']}`",
            f"- infos: `{payload['provisional_gap_ui_state']['infos']}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
