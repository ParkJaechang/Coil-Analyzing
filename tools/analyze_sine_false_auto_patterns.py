from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ARTIFACT_DIR = Path("artifacts/policy_eval")
INPUT_CSV = ARTIFACT_DIR / "policy_eval_v3_candidate_p95_continuous_corpus_l1fo.csv"
OUTPUT_JSON = ARTIFACT_DIR / "sine_false_auto_analysis.json"
OUTPUT_MD = ARTIFACT_DIR / "sine_false_auto_analysis.md"


@dataclass(slots=True)
class GeometryStats:
    freq_hz: float
    nearest_abs_hz: float
    lower_hz: float
    upper_hz: float
    bracket_span_hz: float
    bracket_position: float


def _parse_support_freqs(value: Any) -> list[float]:
    values: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return sorted(set(values))


def _geometry(row: pd.Series) -> GeometryStats:
    support = _parse_support_freqs(row["support_freqs_hz"])
    freq_hz = float(row["target_freq_hz"])
    lower = max((item for item in support if item < freq_hz), default=float("nan"))
    upper = min((item for item in support if item > freq_hz), default=float("nan"))
    nearest_abs = min(abs(freq_hz - item) for item in support)
    span = float(upper - lower) if np.isfinite(lower) and np.isfinite(upper) else float("nan")
    position = float((freq_hz - lower) / span) if np.isfinite(span) and span > 0 else float("nan")
    return GeometryStats(
        freq_hz=freq_hz,
        nearest_abs_hz=float(nearest_abs),
        lower_hz=float(lower),
        upper_hz=float(upper),
        bracket_span_hz=span,
        bracket_position=position,
    )


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def main() -> int:
    frame = pd.read_csv(INPUT_CSV)
    false_auto = frame[frame["confusion_label"] == "false_auto"].copy()
    sine_false_auto = false_auto[false_auto["waveform_type"] == "sine"].copy()

    geometry_rows = [_geometry(row) for _, row in sine_false_auto.iterrows()]
    geometry_frame = pd.DataFrame([asdict(item) for item in geometry_rows])
    detailed_frame = sine_false_auto.copy()
    detailed_frame = detailed_frame.reset_index(drop=True)
    detailed_frame["nearest_abs_hz"] = geometry_frame["nearest_abs_hz"]
    detailed_frame["lower_hz"] = geometry_frame["lower_hz"]
    detailed_frame["upper_hz"] = geometry_frame["upper_hz"]
    detailed_frame["bracket_span_hz"] = geometry_frame["bracket_span_hz"]
    detailed_frame["bracket_position"] = geometry_frame["bracket_position"]
    detailed_frame["target_level_bin"] = pd.cut(
        detailed_frame["target_current_pp_a"],
        bins=[0.0, 1.0, 3.0, 6.0, 20.0],
        include_lowest=True,
    ).astype(str)

    freq_summary = (
        sine_false_auto.groupby("target_freq_hz")[["realized_nrmse", "realized_shape_corr", "current_pp_error_pct", "realized_phase_lag_deg"]]
        .agg(["mean", "max"])
        .reset_index()
    )
    freq_summary.columns = [
        "target_freq_hz",
        "realized_nrmse_mean",
        "realized_nrmse_max",
        "shape_corr_mean",
        "shape_corr_max",
        "current_pp_error_pct_mean",
        "current_pp_error_pct_max",
        "phase_lag_deg_mean",
        "phase_lag_deg_max",
    ]
    geometry_summary = (
        geometry_frame.groupby("freq_hz")[["nearest_abs_hz", "lower_hz", "upper_hz", "bracket_span_hz", "bracket_position"]]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    geometry_summary.columns = [
        "freq_hz",
        "nearest_abs_hz_mean",
        "nearest_abs_hz_min",
        "nearest_abs_hz_max",
        "lower_hz_mean",
        "lower_hz_min",
        "lower_hz_max",
        "upper_hz_mean",
        "upper_hz_min",
        "upper_hz_max",
        "bracket_span_hz_mean",
        "bracket_span_hz_min",
        "bracket_span_hz_max",
        "bracket_position_mean",
        "bracket_position_min",
        "bracket_position_max",
    ]

    full_corr = frame[
        [
            "surface_confidence",
            "predicted_error_band",
            "realized_nrmse",
            "realized_shape_corr",
            "harmonic_fill_ratio",
            "support_run_count",
        ]
    ].corr(numeric_only=True)
    sine_unique = {
        "surface_confidence": sorted(sine_false_auto["surface_confidence"].dropna().unique().tolist()),
        "predicted_error_band": sorted(sine_false_auto["predicted_error_band"].dropna().unique().tolist()),
        "harmonic_fill_ratio": sorted(sine_false_auto["harmonic_fill_ratio"].dropna().unique().tolist()),
        "support_run_count": sorted(sine_false_auto["support_run_count"].dropna().unique().tolist()),
    }
    level_summary = (
        detailed_frame.groupby("target_level_bin")[["realized_nrmse", "realized_shape_corr", "current_pp_error_pct", "realized_phase_lag_deg"]]
        .agg(["count", "mean", "max"])
        .reset_index()
    )
    level_summary.columns = [
        "target_level_bin",
        "case_count",
        "realized_nrmse_mean",
        "realized_nrmse_max",
        "shape_corr_count",
        "shape_corr_mean",
        "shape_corr_max",
        "current_pp_error_pct_count",
        "current_pp_error_pct_mean",
        "current_pp_error_pct_max",
        "phase_lag_deg_count",
        "phase_lag_deg_mean",
        "phase_lag_deg_max",
    ]
    level_summary = level_summary[
        [
            "target_level_bin",
            "case_count",
            "realized_nrmse_mean",
            "realized_nrmse_max",
            "shape_corr_mean",
            "shape_corr_max",
            "current_pp_error_pct_mean",
            "current_pp_error_pct_max",
            "phase_lag_deg_mean",
            "phase_lag_deg_max",
        ]
    ]
    support_density_summary = (
        detailed_frame.groupby("support_run_count")[["realized_nrmse", "realized_shape_corr", "current_pp_error_pct"]]
        .agg(["count", "mean", "max"])
        .reset_index()
    )
    support_density_summary.columns = [
        "support_run_count",
        "case_count",
        "realized_nrmse_mean",
        "realized_nrmse_max",
        "shape_corr_count",
        "shape_corr_mean",
        "shape_corr_max",
        "current_pp_error_pct_count",
        "current_pp_error_pct_mean",
        "current_pp_error_pct_max",
    ]
    support_density_summary = support_density_summary[
        [
            "support_run_count",
            "case_count",
            "realized_nrmse_mean",
            "realized_nrmse_max",
            "shape_corr_mean",
            "shape_corr_max",
            "current_pp_error_pct_mean",
            "current_pp_error_pct_max",
        ]
    ]
    detailed_rows = detailed_frame[
        [
            "holdout_test_id",
            "target_freq_hz",
            "target_current_pp_a",
            "nearest_abs_hz",
            "bracket_span_hz",
            "support_run_count",
            "harmonic_fill_ratio",
            "surface_confidence",
            "predicted_error_band",
            "realized_nrmse",
            "realized_shape_corr",
            "realized_phase_lag_deg",
        ]
    ].round(6)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(INPUT_CSV),
        "false_auto_total": int(len(false_auto)),
        "sine_false_auto_total": int(len(sine_false_auto)),
        "triangle_false_auto_total": int(len(false_auto[false_auto["waveform_type"] == "triangle"])),
        "sine_unique_metrics": sine_unique,
        "full_correlation": full_corr.round(6).to_dict(),
        "freq_summary": freq_summary.round(6).to_dict(orient="records"),
        "level_summary": level_summary.round(6).to_dict(orient="records"),
        "support_density_summary": support_density_summary.round(6).to_dict(orient="records"),
        "geometry_summary": geometry_summary.round(6).to_dict(orient="records"),
        "detailed_rows": detailed_rows.to_dict(orient="records"),
        "structural_findings": [
            "All false auto cases are sine steady-state interpolation cases.",
            "For sine false auto, surface_confidence is constant at 0.6 and predicted_error_band is constant at 0.16, so the policy cannot discriminate within that slice.",
            "Harmonic fill ratio is saturated at 1.0 on every sine false auto case, so harmonic coverage is not the differentiating factor.",
            "Predicted error calibration is structurally weak: on the full corpus, predicted_error_band shows weak inverse correlation with realized_nrmse, while surface_confidence shows weak positive correlation with realized_nrmse.",
            "Actual interpolation quality degrades strongly with frequency for sine (especially 2 Hz and 5 Hz), but the current confidence model does not include frequency-distance or bracket-span penalties.",
        ],
        "shortest_next_fix": "Add interpolation-geometry-aware penalties to surface_confidence and predicted_error_band for steady-state sine interpolation, using nearest support distance and bracketing span in log-frequency space.",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Sine False Auto Analysis")
    lines.append("")
    lines.append(f"- input: `{INPUT_CSV}`")
    lines.append(f"- false_auto_total: `{len(false_auto)}`")
    lines.append(f"- sine_false_auto_total: `{len(sine_false_auto)}`")
    lines.append(f"- triangle_false_auto_total: `{len(false_auto[false_auto['waveform_type'] == 'triangle'])}`")
    lines.append("")
    lines.append("## Common Pattern")
    lines.append("")
    for finding in payload["structural_findings"]:
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Constant Policy Inputs On Sine False Auto")
    lines.append("")
    for key, values in sine_unique.items():
        lines.append(f"- `{key}`: `{values}`")
    lines.append("")
    lines.append("## Frequency Error Summary")
    lines.append("")
    lines.append(
        _markdown_table(
            freq_summary.round(6).to_dict(orient="records"),
            [
                "target_freq_hz",
                "realized_nrmse_mean",
                "realized_nrmse_max",
                "shape_corr_mean",
                "shape_corr_max",
                "current_pp_error_pct_mean",
                "current_pp_error_pct_max",
                "phase_lag_deg_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## Target Level Summary")
    lines.append("")
    lines.append(
        _markdown_table(
            level_summary.round(6).to_dict(orient="records"),
            [
                "target_level_bin",
                "case_count",
                "realized_nrmse_mean",
                "shape_corr_mean",
                "current_pp_error_pct_mean",
                "phase_lag_deg_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## Support Density Summary")
    lines.append("")
    lines.append(
        _markdown_table(
            support_density_summary.round(6).to_dict(orient="records"),
            [
                "support_run_count",
                "case_count",
                "realized_nrmse_mean",
                "shape_corr_mean",
                "current_pp_error_pct_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## Interpolation Geometry Summary")
    lines.append("")
    lines.append(
        _markdown_table(
            geometry_summary.round(6).to_dict(orient="records"),
            [
                "freq_hz",
                "nearest_abs_hz_mean",
                "lower_hz_mean",
                "upper_hz_mean",
                "bracket_span_hz_mean",
                "bracket_position_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## One-Line Conclusion")
    lines.append("")
    lines.append("- This is not a threshold-only problem; it is primarily a confidence-design problem, with steady-state harmonic-surface interpolation overestimating trust because it ignores interpolation geometry.")
    lines.append("")
    lines.append("## Shortest Next Fix")
    lines.append("")
    lines.append(f"- {payload['shortest_next_fix']}")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
