from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


OFFICIAL_MAX_FREQ_HZ = 5.0
POLICY_DIR = Path("artifacts/policy_eval")
EXPORT_JSON = POLICY_DIR / "exact_export_validation.json"
UI_JSON = POLICY_DIR / "ui_policy_validation.json"
AXIS_JSON = POLICY_DIR / "field_axis_sanity.json"
OUTPUT_JSON = POLICY_DIR / "field_exact_promotion_status.json"
OUTPUT_MD = POLICY_DIR / "field_exact_promotion_status.md"


def main() -> int:
    export_validation = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
    ui_validation = json.loads(UI_JSON.read_text(encoding="utf-8"))
    axis_sanity = json.loads(AXIS_JSON.read_text(encoding="utf-8"))

    field_export = export_validation["cases"]["continuous_exact_field"]
    field_ui = next(case for case in ui_validation["cases"] if case["name"] == "exact_field_auto")
    metrics = field_ui.get("metrics", {})

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_support_band_hz": {"min": 0.25, "max": OFFICIAL_MAX_FREQ_HZ},
        "status": "software_ready_bench_pending",
        "official_scope": f"continuous/field exact support only, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz",
        "export_validation": {
            "support_state": field_export["support_state"],
            "preview_only": field_export["preview_only"],
            "allow_auto_download": field_export["allow_auto_download"],
            "lut_voltage_within_waveform_envelope": field_export["lut_voltage_within_waveform_envelope"],
            "lut_cycle_progress_in_unit_interval": field_export["lut_cycle_progress_in_unit_interval"],
        },
        "ui_validation": {
            "pass_status": field_ui["pass_status"],
            "warnings": field_ui["warnings"],
            "successes": field_ui["successes"],
            "metrics": metrics,
        },
        "axis_sanity": axis_sanity,
        "remaining_gate": "bench smoke test sign-off",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Field Exact Promotion Status",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- official_support_band_hz: `0.25 ~ {OFFICIAL_MAX_FREQ_HZ:g}`",
        f"- status: `{payload['status']}`",
        f"- official_scope: `{payload['official_scope']}`",
        "",
        "## Export Validation",
        "",
        f"- support_state: `{field_export['support_state']}`",
        f"- preview_only: `{field_export['preview_only']}`",
        f"- allow_auto_download: `{field_export['allow_auto_download']}`",
        f"- lut_voltage_within_waveform_envelope: `{field_export['lut_voltage_within_waveform_envelope']}`",
        f"- lut_cycle_progress_in_unit_interval: `{field_export['lut_cycle_progress_in_unit_interval']}`",
        "",
        "## UI Validation",
        "",
        f"- pass_status: `{field_ui['pass_status']}`",
        f"- warnings: `{len(field_ui.get('warnings', []))}`",
        f"- successes: `{len(field_ui.get('successes', []))}`",
        "",
        "## Axis Sanity",
        "",
        f"- run_count: `{axis_sanity.get('run_count')}`",
        f"- primary_field_axis_counts: `{axis_sanity.get('primary_field_axis_counts')}`",
        f"- missing_bz_runs: `{axis_sanity.get('missing_bz_runs')}`",
        "",
        "## Remaining Gate",
        "",
        "- bench smoke test sign-off only",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
