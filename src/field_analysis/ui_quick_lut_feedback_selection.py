from __future__ import annotations

from .finite_actual_drive import parse_finite_actual_drive_filename


def classify_feedback_csv_candidate(filename: str, csv_bytes: bytes | None) -> dict[str, object]:
    header = _first_csv_header(csv_bytes)
    columns = {part.strip() for part in header.split(",") if part.strip()}
    if {"sample_index", "time_s", "voltage_v"}.issubset(columns):
        return {
            "file_type": "final_voltage_lut",
            "schema_status": "final_voltage_lut_not_actual_drive_result",
            "message": (
                "이 파일은 최종 전압 LUT CSV입니다. 피드백 보정에는 TimeMs / Voltage1_V / HallBz 컬럼이 있는 "
                "장비 측정 CSV가 필요합니다."
            ),
        }
    try:
        meta = parse_finite_actual_drive_filename(filename)
        if not columns or {"TimeMs", "Voltage1_V", "HallBz"}.issubset(columns):
            return {"file_type": "actual_drive_result", "schema_status": "filename_match", "metadata_source": "filename", **meta}
    except ValueError:
        pass
    if {"TimeMs", "Voltage1_V", "HallBz"}.issubset(columns):
        preamble = _preamble_metadata(csv_bytes)
        if preamble.get("freq_hz") is not None and preamble.get("cycle_count") is not None:
            return {
                "file_type": "actual_drive_result",
                "schema_status": "actual_drive_schema_with_preamble_metadata",
                "metadata_source": "preamble",
                "waveform_type": preamble.get("waveform_type"),
                "freq_hz": preamble.get("freq_hz"),
                "cycle_count": preamble.get("cycle_count"),
            }
        return {
            "file_type": "actual_drive_result",
            "schema_status": "actual_drive_schema_no_filename_metadata",
            "metadata_source": "unavailable",
            "waveform_type": None,
            "freq_hz": None,
            "cycle_count": None,
        }
    return {
        "file_type": "unknown",
        "schema_status": "unsupported_feedback_file",
        "message": "실구동 결과 CSV가 없습니다. TimeMs / Voltage1_V / HallBz 컬럼이 있는 측정 결과를 업로드하십시오.",
    }


def choose_actual_drive_feedback_candidate(
    candidates: list[dict[str, object]],
    *,
    waveform_type: str | None,
    freq_hz: float | None,
    cycle_count: float | None,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    actual_candidates: list[dict[str, object]] = []
    final_lut_count = 0
    for candidate in candidates:
        info = classify_feedback_csv_candidate(
            str(candidate.get("filename") or candidate.get("original_filename") or ""),
            candidate.get("csv_bytes") if isinstance(candidate.get("csv_bytes"), bytes) else None,
        )
        enriched = {**candidate, **info}
        if info.get("file_type") == "actual_drive_result":
            actual_candidates.append(enriched)
        elif info.get("file_type") == "final_voltage_lut":
            final_lut_count += 1

    exact = [
        candidate
        for candidate in actual_candidates
        if candidate_matches(candidate, waveform_type=waveform_type, freq_hz=freq_hz, cycle_count=cycle_count)
    ]
    if len(exact) == 1:
        return exact[0], {"selection_status": "auto_selected", "selection_reason": "exact_match", "candidate_count": len(actual_candidates)}
    if len(exact) > 1:
        return None, {
            "selection_status": "needs_manual_selection",
            "selection_reason": "multiple_exact_matches",
            "candidate_count": len(actual_candidates),
            "exact_match_count": len(exact),
        }
    if len(actual_candidates) == 1:
        return None, {
            "selection_status": "needs_manual_selection",
            "selection_reason": "single_candidate_mismatch_raw_preview_only",
            "candidate_count": 1,
            "warning": "현재 target과 실구동 결과 파일의 주파수/cycle이 일치하지 않아 2차 모델링에 사용할 수 없습니다. 해당 주파수/cycle로 실제 구동한 result CSV를 업로드하십시오.",
        }
    if not actual_candidates and final_lut_count:
        return None, {
            "selection_status": "unavailable",
            "selection_reason": "final_voltage_lut_not_actual_drive_result",
            "candidate_count": 0,
            "final_lut_count": final_lut_count,
        }
    return None, {"selection_status": "unavailable", "selection_reason": "no_actual_drive_result", "candidate_count": len(actual_candidates)}


def candidate_matches(candidate: dict[str, object], *, waveform_type: str | None, freq_hz: float | None, cycle_count: float | None) -> bool:
    if waveform_type and str(candidate.get("waveform_type") or candidate.get("waveform") or "").lower() != str(waveform_type).lower():
        return False
    if freq_hz is not None and candidate.get("freq_hz") is not None:
        if abs(float(candidate["freq_hz"]) - float(freq_hz)) > 1e-9:
            return False
    elif freq_hz is not None:
        return False
    if cycle_count is not None and candidate.get("cycle_count") is not None:
        if abs(float(candidate["cycle_count"]) - float(cycle_count)) > 1e-9:
            return False
    elif cycle_count is not None:
        return False
    return True


def _first_csv_header(csv_bytes: bytes | None) -> str:
    if not csv_bytes:
        return ""
    text = bytes(csv_bytes).decode("utf-8-sig", errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def _preamble_metadata(csv_bytes: bytes | None) -> dict[str, object]:
    if not csv_bytes:
        return {}
    text = bytes(csv_bytes).decode("utf-8-sig", errors="ignore")
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped[1:].split(",")]
        if len(parts) >= 2 and parts[0]:
            values[parts[0]] = parts[1]
    return {
        "waveform_type": (values.get("Waveform") or values.get("WaveformFamily") or "triangle").lower(),
        "freq_hz": _positive_float_or_none(values.get("Frequency(Hz)")),
        "cycle_count": _positive_float_or_none(values.get("Cycles")),
    }


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _positive_float_or_none(value: object) -> float | None:
    number = _float_or_none(value)
    return number if number is not None and number > 0.0 else None
