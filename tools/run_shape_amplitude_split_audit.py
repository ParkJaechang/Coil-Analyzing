from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
SRC_ROOT = REPO_ROOT.parent / "src"
for candidate in (TOOLS_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_release_candidate_real_validations as validation_runtime  # noqa: E402
from field_analysis.recommendation_legacy_bridge import build_finite_support_entries  # noqa: E402
from field_analysis.shape_amplitude_split import (  # noqa: E402
    build_amplitude_lut_audit,
    build_same_freq_level_sensitivity,
    build_shape_engine_audit,
    build_support_route_level_influence,
)


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "bz_first_exact_matrix"
AMPLITUDE_LUT_AUDIT_JSON = ARTIFACT_DIR / "amplitude_lut_audit.json"
AMPLITUDE_LUT_AUDIT_MD = ARTIFACT_DIR / "amplitude_lut_audit.md"
SHAPE_ENGINE_AUDIT_JSON = ARTIFACT_DIR / "shape_engine_audit.json"
SHAPE_ENGINE_AUDIT_MD = ARTIFACT_DIR / "shape_engine_audit.md"
SAME_FREQ_LEVEL_JSON = ARTIFACT_DIR / "same_freq_level_sensitivity.json"
SAME_FREQ_LEVEL_MD = ARTIFACT_DIR / "same_freq_level_sensitivity.md"
SUPPORT_ROUTE_LEVEL_JSON = ARTIFACT_DIR / "support_route_level_influence.json"
SUPPORT_ROUTE_LEVEL_MD = ARTIFACT_DIR / "support_route_level_influence.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _pick_levels(values: list[float], *, count: int = 3) -> list[float]:
    ordered = sorted({float(value) for value in values})
    if len(ordered) <= count:
        return ordered
    mid_index = len(ordered) // 2
    return sorted({ordered[0], ordered[mid_index], ordered[-1]})


def _run_probe(
    *,
    waveform: str,
    target_type: str,
    freq_hz: float,
    target_level: float,
    finite_cycle_mode: bool = False,
    target_cycle_count: float | None = None,
) -> dict[str, Any]:
    result, route_summary, _runtime = validation_runtime.run_case(
        waveform=waveform,
        target_type=target_type,
        freq_hz=freq_hz,
        target_level=target_level,
        finite_cycle_mode=finite_cycle_mode,
        target_cycle_count=target_cycle_count,
    )
    snapshot = validation_runtime._runtime_prediction_snapshot(result)
    return {
        "target_level_pp": float(target_level),
        "route_reason": str(route_summary.get("reason") or ""),
        "support_state": str(result.engine_summary.get("support_state") or ""),
        "amplitude_lut_meaning": result.debug_info.get("amplitude_lut_meaning"),
        "shape_engine_source": result.debug_info.get("shape_engine_source"),
        "amplitude_engine_source": result.debug_info.get("amplitude_engine_source"),
        "pp_affects_shape": result.debug_info.get("pp_affects_shape"),
        **snapshot,
    }


def _write_amplitude_lut_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Amplitude LUT Audit",
        "",
        f"- amplitude_lut_meaning: `{payload.get('amplitude_lut_meaning')}`",
        f"- observed_behavior_classification: `{payload.get('observed_behavior_classification')}`",
        f"- group_count: `{payload.get('summary', {}).get('group_count')}`",
        f"- template_switch_group_count: `{payload.get('summary', {}).get('template_switch_group_count')}`",
        f"- mixed_observed_group_count: `{payload.get('summary', {}).get('mixed_observed_group_count')}`",
        "",
    ]
    for group in payload.get("groups", []):
        lines.extend(
            [
                f"## {group['waveform_type']} / {group['freq_hz']} Hz / {group['target_metric']}",
                f"- implementation_meaning: `{group.get('implementation_meaning')}`",
                f"- observed_shape_meaning: `{group.get('observed_shape_meaning')}`",
                f"- pp_influences_template_selection: `{group.get('pp_influences_template_selection')}`",
                f"- template_test_ids: `{group.get('template_test_ids', [])}`",
                "",
            ]
        )
    _write_markdown(AMPLITUDE_LUT_AUDIT_MD, lines)


def _write_shape_engine_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Shape Engine Audit",
        "",
        f"- shape_engine_source: `{payload.get('prototype', {}).get('shape_engine_source')}`",
        f"- group_count: `{payload.get('summary', {}).get('group_count')}`",
        f"- stable_group_count: `{payload.get('summary', {}).get('stable_group_count')}`",
        f"- unstable_group_count: `{payload.get('summary', {}).get('unstable_group_count')}`",
        "",
    ]
    for group in payload.get("groups", []):
        lines.extend(
            [
                f"## {group['regime']} / {group['waveform_type']} / {group['freq_hz']} Hz",
                f"- pp_affects_shape: `{group.get('pp_affects_shape')}`",
                f"- normalized_bz_shape_preview: `{group.get('normalized_bz_shape_preview', [])[:8]}`",
                "",
            ]
        )
    _write_markdown(SHAPE_ENGINE_AUDIT_MD, lines)


def _write_same_freq_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Same Freq Level Sensitivity",
        "",
        f"- continuous_group_count: `{payload.get('summary', {}).get('continuous_group_count')}`",
        f"- finite_group_count: `{payload.get('summary', {}).get('finite_group_count')}`",
        f"- stable_group_count: `{payload.get('summary', {}).get('stable_group_count')}`",
        f"- unstable_group_count: `{payload.get('summary', {}).get('unstable_group_count')}`",
        "",
    ]
    for group in payload.get("continuous_exact", []) + payload.get("finite_exact", []):
        lines.extend(
            [
                f"## {group['regime']} / {group['waveform_type']} / {group['freq_hz']} Hz",
                f"- pp_affects_shape: `{group.get('pp_affects_shape')}`",
                f"- signal_summaries: `{sorted(group.get('signal_summaries', {}).keys())}`",
                "",
            ]
        )
    _write_markdown(SAME_FREQ_LEVEL_MD, lines)


def _write_support_route_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Support Route Level Influence",
        "",
        f"- group_count: `{payload.get('summary', {}).get('group_count')}`",
        f"- shape_affected_group_count: `{payload.get('summary', {}).get('shape_affected_group_count')}`",
        f"- support_switch_group_count: `{payload.get('summary', {}).get('support_switch_group_count')}`",
        f"- source_switch_group_count: `{payload.get('summary', {}).get('source_switch_group_count')}`",
        "",
    ]
    for group in payload.get("probe_groups", []):
        lines.extend(
            [
                f"## {group['probe_group']}",
                f"- levels: `{group.get('levels')}`",
                f"- pp_affects_shape: `{group.get('pp_affects_shape')}`",
                f"- reason_codes: `{group.get('reason_codes', [])}`",
                f"- selected_support_ids: `{group.get('selected_support_ids', [])}`",
                f"- prediction_sources: `{group.get('prediction_sources', [])}`",
                "",
            ]
        )
    _write_markdown(SUPPORT_ROUTE_LEVEL_MD, lines)


def main() -> int:
    runtime = validation_runtime._load_backend_probe_runtime()
    legacy_context = runtime.legacy_context
    summary = legacy_context.per_test_summary

    finite_entries = build_finite_support_entries(
        transient_measurements=legacy_context.transient_measurements,
        transient_preprocess_results=legacy_context.transient_preprocess_results,
        transient_canonical_runs=legacy_context.transient_canonical_runs,
        current_channel=validation_runtime.CURRENT_CHANNEL,
        field_channel=validation_runtime.FIELD_CHANNEL,
    )

    amplitude_lut_audit = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        **build_amplitude_lut_audit(
            per_test_summary=summary,
            analyses_by_test_id=legacy_context.analysis_lookup,
        ),
    }
    shape_engine_audit = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        **build_shape_engine_audit(
            analyses_by_test_id=legacy_context.analysis_lookup,
            per_test_summary=summary,
            finite_entries=finite_entries,
            current_channel=validation_runtime.CURRENT_CHANNEL,
            field_channel=validation_runtime.FIELD_CHANNEL,
        ),
    }
    same_freq_level_sensitivity = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        **build_same_freq_level_sensitivity(shape_engine_audit=shape_engine_audit),
    }

    current_levels = _pick_levels(
        summary[
            (summary["waveform_type"] == "sine")
            & (pd.to_numeric(summary["freq_hz"], errors="coerce").sub(0.5).abs() <= 1e-9)
        ]["current_pp_target_a"].dropna().tolist()
    )
    field_support_levels = _pick_levels(
        summary[
            (summary["waveform_type"] == "sine")
            & (pd.to_numeric(summary["freq_hz"], errors="coerce").sub(0.25).abs() <= 1e-9)
        ]["achieved_bz_mT_pp_mean"].dropna().tolist()
    )
    field_levels = sorted({20.0, *field_support_levels})
    finite_levels = _pick_levels(
        [
            value
            for value in (
                pd.to_numeric(pd.DataFrame(finite_entries).get("requested_current_pp"), errors="coerce")
                if finite_entries
                else pd.Series(dtype=float)
            ).dropna().tolist()
        ]
    )

    probe_groups = {
        "continuous_current_exact_sine_0p5hz": [
            _run_probe(waveform="sine", target_type="current", freq_hz=0.5, target_level=level)
            for level in current_levels
        ],
        "continuous_field_exact_sine_0p25hz": [
            _run_probe(waveform="sine", target_type="field", freq_hz=0.25, target_level=level)
            for level in field_levels
        ],
        "finite_exact_triangle_1p25hz_1p25cycle": [
            _run_probe(
                waveform="triangle",
                target_type="current",
                freq_hz=1.25,
                target_level=level,
                finite_cycle_mode=True,
                target_cycle_count=1.25,
            )
            for level in finite_levels
        ],
    }
    support_route_level_influence = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        **build_support_route_level_influence(probe_groups=probe_groups),
    }

    _write_json(AMPLITUDE_LUT_AUDIT_JSON, amplitude_lut_audit)
    _write_json(SHAPE_ENGINE_AUDIT_JSON, shape_engine_audit)
    _write_json(SAME_FREQ_LEVEL_JSON, same_freq_level_sensitivity)
    _write_json(SUPPORT_ROUTE_LEVEL_JSON, support_route_level_influence)
    _write_amplitude_lut_markdown(amplitude_lut_audit)
    _write_shape_engine_markdown(shape_engine_audit)
    _write_same_freq_markdown(same_freq_level_sensitivity)
    _write_support_route_markdown(support_route_level_influence)

    summary_payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "amplitude_lut_audit": AMPLITUDE_LUT_AUDIT_JSON.as_posix(),
        "shape_engine_audit": SHAPE_ENGINE_AUDIT_JSON.as_posix(),
        "same_freq_level_sensitivity": SAME_FREQ_LEVEL_JSON.as_posix(),
        "support_route_level_influence": SUPPORT_ROUTE_LEVEL_JSON.as_posix(),
        "amplitude_lut_summary": amplitude_lut_audit.get("summary", {}),
        "shape_engine_summary": shape_engine_audit.get("summary", {}),
        "same_freq_summary": same_freq_level_sensitivity.get("summary", {}),
        "support_route_summary": support_route_level_influence.get("summary", {}),
    }
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
