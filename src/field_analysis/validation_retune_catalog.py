from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

from .validation_retune import (
    REQUIRED_CORRECTED_ARTIFACT_KEYS,
    SOURCE_KIND_CORRECTED,
    SOURCE_KIND_EXPORT,
    SOURCE_KIND_RECOMMENDATION,
    SOURCE_KIND_UNKNOWN,
    build_quality_badge_markdown,
    normalize_corrected_lineage_root,
    parse_corrected_iteration_index,
    to_jsonable,
)
from .runtime_display_labels import (
    build_display_label,
    build_display_name,
    build_display_object_key,
    infer_iteration_index,
    sanitize_display_text,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def build_validation_catalog_payload(report_dirs: list[Path], history_path: Path) -> dict[str, Any]:
    entries_by_key: dict[str, dict[str, Any]] = {}

    def merge_entry(entry: dict[str, Any]) -> None:
        key = str(entry.get("validation_run_id") or entry.get("corrected_lut_id") or entry.get("report_path") or len(entries_by_key))
        existing = entries_by_key.get(key)
        if existing is None:
            entries_by_key[key] = entry
            return
        for field, value in entry.items():
            if field == "quality_reasons":
                combined = list(existing.get(field) or [])
                if isinstance(value, list):
                    combined.extend(str(item) for item in value if item)
                existing[field] = list(dict.fromkeys(combined))
                continue
            if existing.get(field) in (None, "", [], {}):
                existing[field] = value

    for report_dir in report_dirs:
        if not report_dir.exists():
            continue
        for report_path in sorted(report_dir.glob("*_validation_report.json")):
            entry = _normalize_validation_report(report_path)
            if entry is not None:
                merge_entry(entry)

    history = load_json(history_path)
    for item in history.get("retunes", []):
        if isinstance(item, dict):
            entry = _normalize_history_entry(item)
            if entry is not None:
                merge_entry(entry)

    entries = list(entries_by_key.values())
    _annotate_validation_entries(entries)
    _annotate_display_fields(entries)
    entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {
        "generated_at": utc_now(),
        "schema_version": "validation_catalog_v3",
        "summary": _catalog_summary(entries),
        "filters": _catalog_filters(entries),
        "entries": entries,
    }


def build_corrected_lut_catalog_payload(validation_entries: list[dict[str, Any]]) -> dict[str, Any]:
    corrected_entries = [
        {
            "corrected_lut_id": item.get("corrected_lut_id"),
            "original_recommendation_id": item.get("original_recommendation_id"),
            "lut_id": item.get("lut_id"),
            "validation_run_id": item.get("validation_run_id"),
            "source_kind": item.get("source_kind"),
            "source_lut_filename": item.get("source_lut_filename"),
            "iteration_index": item.get("iteration_index"),
            "created_at": item.get("created_at"),
            "exact_path": item.get("exact_path"),
            "target_output_type": item.get("target_output_type"),
            "waveform_type": item.get("waveform_type"),
            "freq_hz": item.get("freq_hz"),
            "commanded_cycles": item.get("commanded_cycles"),
            "quality_label": item.get("quality_label"),
            "quality_tone": item.get("quality_tone"),
            "quality_reasons": item.get("quality_reasons", []),
            "acceptance_decision": item.get("acceptance_decision"),
            "candidate_status": item.get("candidate_status"),
            "candidate_status_label": item.get("candidate_status_label"),
            "preferred_output_id": item.get("preferred_output_id"),
            "preferred_output_kind": item.get("preferred_output_kind"),
            "rejection_reason": item.get("rejection_reason"),
            "lineage_root_id": item.get("lineage_root_id"),
            "display_object_key": item.get("display_object_key"),
            "display_name": item.get("display_name"),
            "display_label": build_display_label(
                display_name=str(item.get("display_name") or ""),
                source_kind=SOURCE_KIND_CORRECTED,
                iteration_index=int(item.get("iteration_index")) if safe_float(item.get("iteration_index")) is not None else infer_iteration_index(item.get("corrected_lut_id")),
                include_source_context=True,
            ),
            "source_lut_display_name": item.get("source_lut_display_name"),
            "validation_test_display_name": item.get("validation_test_display_name"),
            "latest_corrected_candidate": item.get("latest_corrected_candidate"),
            "latest_corrected_candidate_id": item.get("latest_corrected_candidate_id"),
            "duplicate": item.get("duplicate"),
            "stale": item.get("stale"),
            "status": item.get("status"),
            "artifact_complete": item.get("artifact_complete"),
            "report_path": item.get("report_path"),
            "artifact_paths": item.get("artifact_paths", {}),
        }
        for item in validation_entries
        if item.get("corrected_lut_id")
    ]
    lineages = _lineage_summary(corrected_entries)
    return {
        "generated_at": utc_now(),
        "schema_version": "corrected_lut_catalog_v3",
        "summary": _catalog_summary(corrected_entries),
        "filters": _catalog_filters(corrected_entries),
        "lineages": lineages,
        "entries": corrected_entries,
    }


def build_retune_picker_payload(
    *,
    lut_entries: list[dict[str, Any]],
    validation_entries: list[dict[str, Any]],
    corrected_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    latest_validation_by_lineage: dict[str, dict[str, Any]] = {}
    for item in sorted(validation_entries, key=lambda entry: str(entry.get("created_at") or ""), reverse=True):
        lineage_root_id = str(
            item.get("lineage_root_id")
            or item.get("original_recommendation_id")
            or item.get("lut_id")
            or item.get("corrected_lut_id")
            or ""
        )
        if lineage_root_id and lineage_root_id not in latest_validation_by_lineage:
            latest_validation_by_lineage[lineage_root_id] = item
    latest_corrected_by_lineage = {
        str(item.get("lineage_root_id")): str(item.get("latest_corrected_candidate_id"))
        for item in corrected_entries
        if item.get("lineage_root_id") and item.get("latest_corrected_candidate_id")
    }

    for item in lut_entries:
        source_kind = str(item.get("catalog_source_kind") or _infer_picker_source_kind(item))
        if source_kind == SOURCE_KIND_CORRECTED:
            continue
        source_id = str(item.get("lut_id") or "")
        if not source_id:
            continue
        selection_id = f"{source_kind}::{source_id}"
        if selection_id in seen:
            continue
        seen.add(selection_id)
        lineage_root_id = str(item.get("original_recommendation_id") or source_id)
        latest_validation = latest_validation_by_lineage.get(lineage_root_id, {})
        entries.append(
            {
                "selection_id": selection_id,
                "source_kind": source_kind,
                "source_id": source_id,
                "display_object_key": item.get("display_object_key"),
                "display_name": item.get("display_name") or _build_picker_name(item),
                "display_label": _build_picker_label(item),
                "retune_eligible": bool(item.get("exact_support")) and str(item.get("route_origin")) == "exact",
                "status": "eligible_exact_source" if bool(item.get("exact_support")) and str(item.get("route_origin")) == "exact" else "not_exact_source",
                "original_recommendation_id": lineage_root_id,
                "current_lut_id": source_id,
                "lineage_root_id": lineage_root_id,
                "latest_corrected_candidate_id": latest_corrected_by_lineage.get(
                    lineage_root_id
                ),
                "latest_validation_run_id": latest_validation.get("validation_run_id"),
                "latest_validation_quality_label": latest_validation.get("quality_label"),
                "latest_validation_created_at": latest_validation.get("created_at"),
                "target_output_type": item.get("target_type"),
                "waveform_type": item.get("waveform"),
                "freq_hz": item.get("freq_hz"),
                "commanded_cycles": item.get("cycle_count"),
                "exact_path": item.get("exact_path"),
                "created_at": item.get("created_at"),
                "profile_csv_path": item.get("profile_csv_path"),
                "control_lut_path": item.get("control_lut_path"),
                "source_lut_filename": item.get("file_name"),
                "route_origin": item.get("route_origin"),
            }
        )

    for item in corrected_entries:
        corrected_id = str(item.get("corrected_lut_id") or "")
        if not corrected_id:
            continue
        selection_id = f"{SOURCE_KIND_CORRECTED}::{corrected_id}"
        if selection_id in seen:
            continue
        seen.add(selection_id)
        artifact_paths = item.get("artifact_paths", {})
        lineage_root_id = str(item.get("lineage_root_id") or item.get("original_recommendation_id") or corrected_id)
        latest_validation = latest_validation_by_lineage.get(lineage_root_id, {})
        picker_label_item = {
            "source_kind": SOURCE_KIND_CORRECTED,
            "source_id": corrected_id,
            "waveform_type": item.get("waveform_type"),
            "freq_hz": item.get("freq_hz"),
            "commanded_cycles": item.get("commanded_cycles"),
            "target_output_type": item.get("target_output_type"),
            "target_level_value": item.get("target_level_value"),
            "iteration_index": item.get("iteration_index"),
            "display_name": item.get("display_name"),
            "display_object_key": item.get("display_object_key"),
        }
        entries.append(
            {
                "selection_id": selection_id,
                "source_kind": SOURCE_KIND_CORRECTED,
                "source_id": corrected_id,
                "display_object_key": item.get("display_object_key"),
                "display_name": item.get("display_name") or _build_picker_name(picker_label_item),
                "display_label": _build_picker_label(picker_label_item),
                "retune_eligible": bool(item.get("artifact_complete")) and not bool(item.get("duplicate")) and not bool(item.get("stale")),
                "status": (
                    "eligible_corrected_candidate"
                    if bool(item.get("artifact_complete")) and not bool(item.get("duplicate")) and not bool(item.get("stale"))
                    else str(item.get("status") or "corrected_unavailable")
                ),
                "original_recommendation_id": item.get("original_recommendation_id"),
                "current_lut_id": corrected_id,
                "lineage_root_id": lineage_root_id,
                "validation_run_id": item.get("validation_run_id"),
                "corrected_lut_id": corrected_id,
                "latest_corrected_candidate_id": item.get("latest_corrected_candidate_id") or corrected_id,
                "latest_validation_run_id": latest_validation.get("validation_run_id") or item.get("validation_run_id"),
                "latest_validation_quality_label": latest_validation.get("quality_label") or item.get("quality_label"),
                "latest_validation_created_at": latest_validation.get("created_at") or item.get("created_at"),
                "iteration_index": item.get("iteration_index"),
                "target_output_type": item.get("target_output_type"),
                "waveform_type": item.get("waveform_type"),
                "freq_hz": item.get("freq_hz"),
                "commanded_cycles": item.get("commanded_cycles"),
                "exact_path": item.get("exact_path"),
                "quality_label": item.get("quality_label"),
                "quality_tone": item.get("quality_tone"),
                "quality_reasons": list(item.get("quality_reasons") or []),
                "latest_corrected_candidate": bool(item.get("latest_corrected_candidate")),
                "created_at": item.get("created_at"),
                "profile_csv_path": artifact_paths.get("corrected_waveform_csv"),
                "control_lut_path": artifact_paths.get("corrected_control_lut_csv"),
                "source_lut_filename": item.get("source_lut_filename"),
                "report_path": item.get("report_path"),
            }
        )

    entries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {
        "generated_at": utc_now(),
        "schema_version": "retune_picker_catalog_v2",
        "summary": _catalog_summary(entries),
        "filters": _catalog_filters(entries),
        "contract": {
            "selection_id_format": "<source_kind>::<source_id>",
            "source_kinds": [SOURCE_KIND_RECOMMENDATION, SOURCE_KIND_EXPORT, SOURCE_KIND_CORRECTED],
            "required_fields": [
                "selection_id",
                "source_kind",
                "source_id",
                "display_label",
                "retune_eligible",
                "status",
                "profile_csv_path",
            ],
        },
        "entries": entries,
    }


def build_provenance_badge_markdown(
    *,
    validation_payload: dict[str, Any],
    corrected_payload: dict[str, Any],
    picker_payload: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Provenance / Quality Badge",
            "",
            build_quality_badge_markdown(),
            "",
            "## Provenance Fields",
            "- `original_recommendation_id`: corrected lineage의 루트 recommendation ID",
            "- `validation_run_id`: 실제 validation 입력과 연결되는 run key",
            "- `corrected_lut_id`: corrected LUT 산출물 묶음의 canonical ID",
            "- `source_lut_filename`: retune에 사용한 원본 LUT/export 파일명",
            "- `iteration_index`: corrected lineage 내 반복 차수",
            "- `created_at`: validation / corrected 생성 시각",
            "- `correction_rule`: validation residual retune 규칙 문자열",
            "- `before_after_metrics`: target output / bz_effective 도메인의 before/after 비교",
            "",
            "## Picker Contract",
            "- `selection_id = <source_kind>::<source_id>`",
            "- `source_kind`는 `recommendation`, `export`, `corrected` 중 하나",
            "- `retune_eligible=true` 인 항목만 standalone picker에서 바로 재사용 가능",
            "",
            "## Current Counts",
            f"- validation entries: `{len(validation_payload.get('entries', []))}`",
            f"- corrected LUT entries: `{len(corrected_payload.get('entries', []))}`",
            f"- picker entries: `{len(picker_payload.get('entries', []))}`",
        ]
    ) + "\n"


def _normalize_validation_report(report_path: Path) -> dict[str, Any] | None:
    report = load_json(report_path)
    if report.get("schema_version") not in {"validation_retune_v2"}:
        return None
    provenance = report.get("provenance", {})
    validation_run = report.get("validation_run", {})
    quality = report.get("quality_badge", {})
    acceptance = report.get("acceptance_decision", {})
    before_after = report.get("before_after_metrics", {})
    target_before = before_after.get("target_output", {}).get("before") or report.get("baseline_comparison", {})
    target_after = before_after.get("target_output", {}).get("after") or report.get("corrected_comparison", {})
    bz_before = before_after.get("bz_effective", {}).get("before") or report.get("baseline_bz_comparison", {})
    bz_after = before_after.get("bz_effective", {}).get("after") or report.get("corrected_bz_comparison", {})
    corrected_lut_id = provenance.get("corrected_lut_id") or validation_run.get("corrected_lut_id")
    iteration_index = provenance.get("iteration_index") or validation_run.get("iteration_index") or parse_corrected_iteration_index(corrected_lut_id)
    original_recommendation_id = provenance.get("original_recommendation_id") or validation_run.get("original_recommendation_id")
    lut_id = provenance.get("lut_id") or validation_run.get("lut_id")
    lineage_root_id = normalize_corrected_lineage_root(original_recommendation_id or corrected_lut_id or lut_id)
    artifact_paths = report.get("artifact_paths", {})
    return {
        "report_path": report_path.as_posix(),
        "validation_run_id": provenance.get("validation_run_id") or validation_run.get("validation_run_id"),
        "original_recommendation_id": original_recommendation_id,
        "lut_id": lut_id,
        "corrected_lut_id": corrected_lut_id,
        "source_kind": provenance.get("source_kind") or validation_run.get("source_kind"),
        "source_lut_filename": provenance.get("source_lut_filename") or validation_run.get("source_lut_filename"),
        "source_profile_path": provenance.get("source_profile_path") or validation_run.get("source_profile_path"),
        "target_output_type": validation_run.get("target_output_type"),
        "waveform_type": validation_run.get("waveform_type"),
        "freq_hz": validation_run.get("freq_hz"),
        "commanded_cycles": validation_run.get("commanded_cycles"),
        "target_level_value": validation_run.get("target_level_value"),
        "target_level_kind": validation_run.get("target_level_kind"),
        "validation_test_id": validation_run.get("selected_validation_test_id"),
        "created_at": validation_run.get("created_at"),
        "exact_path": provenance.get("exact_path") or validation_run.get("exact_path"),
        "iteration_index": iteration_index,
        "correction_rule": provenance.get("correction_rule") or validation_run.get("correction_rule"),
        "quality_label": quality.get("label"),
        "quality_tone": quality.get("tone"),
        "quality_reasons": quality.get("reasons", []),
        "acceptance_decision": acceptance or report.get("acceptance_decision"),
        "candidate_status": acceptance.get("decision") or report.get("candidate_status"),
        "candidate_status_label": acceptance.get("label") or report.get("candidate_status_label"),
        "preferred_output_id": acceptance.get("preferred_output_id") or report.get("preferred_output_id"),
        "preferred_output_kind": acceptance.get("preferred_output_kind") or report.get("preferred_output_kind"),
        "rejection_reason": acceptance.get("rejection_reason") or report.get("rejection_reason"),
        "baseline_nrmse": target_before.get("nrmse"),
        "corrected_nrmse": target_after.get("nrmse"),
        "baseline_shape_corr": target_before.get("shape_corr"),
        "corrected_shape_corr": target_after.get("shape_corr"),
        "baseline_bz_nrmse": bz_before.get("nrmse"),
        "corrected_bz_nrmse": bz_after.get("nrmse"),
        "baseline_bz_shape_corr": bz_before.get("shape_corr"),
        "corrected_bz_shape_corr": bz_after.get("shape_corr"),
        "baseline_bz_phase_lag_s": bz_before.get("phase_lag_s"),
        "corrected_bz_phase_lag_s": bz_after.get("phase_lag_s"),
        "artifact_paths": artifact_paths,
        "artifact_complete": all(key in artifact_paths for key in REQUIRED_CORRECTED_ARTIFACT_KEYS),
        "lineage_root_id": lineage_root_id,
    }


def _normalize_history_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    corrected_lut_id = item.get("corrected_lut_id")
    original_recommendation_id = item.get("original_recommendation_id")
    lut_id = item.get("lut_id")
    artifact_paths = item.get("artifact_paths", {}) or {}
    if not corrected_lut_id:
        return None
    report_path = item.get("report_path") or artifact_paths.get("validation_report_json")
    return {
        "report_path": report_path,
        "validation_run_id": item.get("validation_run_id"),
        "original_recommendation_id": original_recommendation_id,
        "lut_id": lut_id,
        "corrected_lut_id": corrected_lut_id,
        "source_kind": item.get("source_kind"),
        "source_lut_filename": item.get("source_lut_filename"),
        "target_output_type": item.get("target_output_type"),
        "waveform_type": item.get("waveform_type"),
        "freq_hz": item.get("freq_hz"),
        "commanded_cycles": item.get("commanded_cycles"),
        "target_level_value": item.get("target_level_value"),
        "target_level_kind": item.get("target_level_kind"),
        "validation_test_id": item.get("validation_test_id"),
        "created_at": item.get("created_at"),
        "exact_path": item.get("exact_path"),
        "iteration_index": item.get("iteration_index") or parse_corrected_iteration_index(corrected_lut_id),
        "correction_rule": item.get("correction_rule"),
        "quality_label": item.get("quality_label"),
        "quality_tone": item.get("quality_tone"),
        "quality_reasons": item.get("quality_reasons", []),
        "acceptance_decision": item.get("acceptance_decision"),
        "candidate_status": item.get("candidate_status"),
        "candidate_status_label": item.get("candidate_status_label"),
        "preferred_output_id": item.get("preferred_output_id"),
        "preferred_output_kind": item.get("preferred_output_kind"),
        "rejection_reason": item.get("rejection_reason"),
        "baseline_nrmse": item.get("before_nrmse"),
        "corrected_nrmse": item.get("after_nrmse"),
        "baseline_shape_corr": item.get("before_shape_corr"),
        "corrected_shape_corr": item.get("after_shape_corr"),
        "artifact_paths": artifact_paths,
        "artifact_complete": all(key in artifact_paths for key in REQUIRED_CORRECTED_ARTIFACT_KEYS),
        "lineage_root_id": normalize_corrected_lineage_root(original_recommendation_id or corrected_lut_id or lut_id),
    }


def _annotate_validation_entries(entries: list[dict[str, Any]]) -> None:
    duplicate_counts = Counter(str(item.get("corrected_lut_id") or "") for item in entries if item.get("corrected_lut_id"))
    lineage_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in entries:
        item["duplicate"] = duplicate_counts.get(str(item.get("corrected_lut_id") or ""), 0) > 1 if item.get("corrected_lut_id") else False
        item["stale"] = False
        item["latest_corrected_candidate"] = False
        item["latest_corrected_candidate_id"] = None
        lineage_groups[str(item.get("lineage_root_id") or "")].append(item)

    for lineage_root, group in lineage_groups.items():
        if not lineage_root:
            continue
        sortable = sorted(
            group,
            key=lambda item: (
                safe_float(item.get("iteration_index")) or 0.0,
                str(item.get("created_at") or ""),
                str(item.get("corrected_lut_id") or ""),
            ),
        )
        latest = next((item for item in reversed(sortable) if item.get("corrected_lut_id")), None)
        latest_id = latest.get("corrected_lut_id") if latest else None
        for item in group:
            item["latest_corrected_candidate_id"] = latest_id
            item["latest_corrected_candidate"] = bool(latest_id) and item.get("corrected_lut_id") == latest_id
            item["stale"] = bool(item.get("corrected_lut_id")) and bool(latest_id) and item.get("corrected_lut_id") != latest_id
            if item.get("duplicate"):
                item["status"] = "duplicate"
            elif not item.get("artifact_complete", True):
                item["status"] = "artifact_incomplete"
            elif item.get("latest_corrected_candidate"):
                item["status"] = "latest_corrected_candidate"
            elif item.get("stale"):
                item["status"] = "stale"
            else:
                item["status"] = "validated"


def _annotate_display_fields(entries: list[dict[str, Any]]) -> None:
    for item in entries:
        display_name = build_display_name(
            target_type=item.get("target_output_type"),
            waveform=item.get("waveform_type"),
            freq_hz=item.get("freq_hz"),
            cycle_count=item.get("commanded_cycles"),
            level=item.get("target_level_value"),
            level_kind=item.get("target_level_kind"),
            fallback_texts=(
                item.get("source_lut_filename"),
                item.get("lut_id"),
                item.get("corrected_lut_id"),
                item.get("original_recommendation_id"),
            ),
        )
        iteration_index = int(item.get("iteration_index")) if safe_float(item.get("iteration_index")) is not None else infer_iteration_index(item.get("corrected_lut_id"))
        item["display_object_key"] = build_display_object_key(
            target_type=item.get("target_output_type"),
            waveform=item.get("waveform_type"),
            freq_hz=item.get("freq_hz"),
            cycle_count=item.get("commanded_cycles"),
            level=item.get("target_level_value"),
            level_kind=item.get("target_level_kind"),
        )
        item["display_name"] = display_name
        item["display_label"] = build_display_label(
            display_name=display_name,
            source_kind=item.get("source_kind"),
            iteration_index=iteration_index,
            status=item.get("status"),
            include_source_context=bool(item.get("corrected_lut_id")),
        )
        item["source_lut_display_name"] = build_display_name(
            target_type=item.get("target_output_type"),
            waveform=item.get("waveform_type"),
            freq_hz=item.get("freq_hz"),
            cycle_count=item.get("commanded_cycles"),
            level=item.get("target_level_value"),
            level_kind=item.get("target_level_kind"),
            fallback_texts=(
                item.get("source_lut_filename"),
                item.get("lut_id"),
                item.get("original_recommendation_id"),
            ),
        )
        item["validation_test_display_name"] = sanitize_display_text(item.get("validation_test_id"))


def _catalog_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(entries),
        "by_status": dict(Counter(str(item.get("status") or "unknown") for item in entries)),
        "by_quality_label": dict(Counter(str(item.get("quality_label") or "unknown") for item in entries if "quality_label" in item)),
        "by_candidate_status": dict(Counter(str(item.get("candidate_status") or "unknown") for item in entries if "candidate_status" in item)),
        "by_target_output_type": dict(Counter(str(item.get("target_output_type") or item.get("target_type") or "unknown") for item in entries)),
        "by_exact_path": dict(Counter(str(item.get("exact_path") or "unknown") for item in entries if "exact_path" in item)),
    }


def _catalog_filters(entries: list[dict[str, Any]]) -> dict[str, Any]:
    def values(field: str) -> list[str]:
        return sorted({str(item.get(field)) for item in entries if item.get(field) not in (None, "")})

    return {
        "status": values("status"),
        "quality_label": values("quality_label"),
        "candidate_status": values("candidate_status"),
        "target_output_type": values("target_output_type"),
        "source_kind": values("source_kind"),
        "exact_path": values("exact_path"),
        "waveform_type": values("waveform_type"),
    }


def _lineage_summary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in entries:
        grouped[str(item.get("lineage_root_id") or "")].append(item)
    rows: list[dict[str, Any]] = []
    for lineage_root, group in sorted(grouped.items()):
        if not lineage_root:
            continue
        latest = next((item for item in group if item.get("latest_corrected_candidate")), None)
        rows.append(
            {
                "lineage_root_id": lineage_root,
                "original_recommendation_id": next((item.get("original_recommendation_id") for item in group if item.get("original_recommendation_id")), None),
                "latest_corrected_candidate_id": latest.get("corrected_lut_id") if latest else None,
                "entry_count": len(group),
                "status_counts": dict(Counter(str(item.get("status") or "unknown") for item in group)),
            }
        )
    return rows


def _infer_picker_source_kind(item: dict[str, Any]) -> str:
    profile_path = str(item.get("profile_csv_path") or "").replace("\\", "/").lower()
    if "/recommendation_library/" in profile_path:
        return SOURCE_KIND_RECOMMENDATION
    if "/validation_retune/" in profile_path:
        return SOURCE_KIND_CORRECTED
    if "/export_validation/" in profile_path or str(item.get("lut_id") or "").startswith("control_formula_"):
        return SOURCE_KIND_EXPORT
    return SOURCE_KIND_UNKNOWN


def _build_picker_label(item: dict[str, Any]) -> str:
    return build_display_label(
        display_name=_build_picker_name(item),
        source_kind=str(item.get("source_kind") or item.get("catalog_source_kind") or _infer_picker_source_kind(item)),
        iteration_index=int(item.get("iteration_index")) if safe_float(item.get("iteration_index")) is not None else infer_iteration_index(item.get("source_id") or item.get("corrected_lut_id")),
        include_source_context=True,
    )


def _build_picker_name(item: dict[str, Any]) -> str:
    existing = sanitize_display_text(item.get("display_name"))
    if existing:
        return existing
    return build_display_name(
        target_type=item.get("target_output_type") or item.get("target_type"),
        waveform=item.get("waveform_type") or item.get("waveform"),
        freq_hz=item.get("freq_hz"),
        cycle_count=item.get("commanded_cycles") or item.get("cycle_count"),
        level=item.get("target_level_value") or item.get("level") or item.get("target_output_pp"),
        level_kind=item.get("target_level_kind"),
        fallback_texts=(
            item.get("source_lut_filename"),
            item.get("source_id"),
            item.get("lut_id"),
            item.get("corrected_lut_id"),
            item.get("file_name"),
        ),
    )
