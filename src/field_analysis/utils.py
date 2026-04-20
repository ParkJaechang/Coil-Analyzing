from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
STRUCTURED_FILENAME_PATTERN = re.compile(
    r"(?:^|[/\\]|[_\-\s])(?:[0-9a-f]{6,}[_\-])?(?P<waveform>sine|sin|triangle|tri)"
    r"[_\-\s]+(?P<freq>\d+(?:[.p]\d+)?)(?:hz)?[_\-\s]+(?P<current>\d+(?:[.p]\d+)?)(?:a|app)?",
    flags=re.IGNORECASE,
)
TRANSIENT_FILENAME_PATTERN = re.compile(
    r"(?:^|[/\\]|[_\-\s])(?:[0-9a-f]{6,}[_\-])?(?P<freq>\d+(?:[.p]\d+)?)hz"
    r"[_\-\s]+(?P<cycle>\d+(?:[.p]\d+)?)cycle[_\-\s]+(?P<current>\d+(?:[.p]\d+)?)(?P<mode>pp|a|app)",
    flags=re.IGNORECASE,
)
WAVEFORM_ALIASES = {
    "sine": {
        "sine",
        "sin",
        "sinwave",
        "sinusoid",
        "sinusoidal",
        "sinusidal",
        "사인",
        "사인파",
        "정현파",
    },
    "triangle": {
        "triangle",
        "tri",
        "trianglewave",
        "triwave",
        "삼각",
        "삼각파",
    },
}

BZ_RAW_COLUMN = "bz_raw_mT"
BZ_EFFECTIVE_COLUMN = "bz_effective_mT"


def normalize_name(value: str) -> str:
    """Normalize text for resilient header matching."""

    cleaned = re.sub(r"[\s_\-/(){}\[\],.:;+]+", "", str(value).strip().lower())
    return cleaned.replace("℃", "c").replace("°c", "c")


def first_number(value: object) -> float | None:
    """Extract the first numeric token from text."""

    if value is None:
        return None
    match = NUMBER_PATTERN.search(str(value))
    if match is None:
        return None
    return float(match.group())


def coerce_float(value: object, default: float | None = None) -> float | None:
    """Convert a value to float, returning default on failure."""

    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_bz_effective_convention(
    frame: pd.DataFrame,
    *,
    raw_column: str = "bz_mT",
    preserved_raw_column: str = BZ_RAW_COLUMN,
    effective_column: str = BZ_EFFECTIVE_COLUMN,
    overwrite_source_column: bool = True,
) -> pd.DataFrame:
    """Fix the global convention `bz_effective = -bz_raw` on a frame in-place."""

    if effective_column in frame.columns:
        effective = pd.to_numeric(frame[effective_column], errors="coerce")
        frame[effective_column] = effective
        if preserved_raw_column not in frame.columns:
            frame[preserved_raw_column] = -effective
        else:
            frame[preserved_raw_column] = pd.to_numeric(frame[preserved_raw_column], errors="coerce")
        if overwrite_source_column and raw_column in frame.columns:
            frame[raw_column] = effective
        elif overwrite_source_column and raw_column not in frame.columns:
            frame[raw_column] = effective
        return frame

    if preserved_raw_column in frame.columns:
        raw = pd.to_numeric(frame[preserved_raw_column], errors="coerce")
    elif raw_column in frame.columns:
        raw = pd.to_numeric(frame[raw_column], errors="coerce")
    else:
        return frame

    effective = -raw
    frame[preserved_raw_column] = raw
    frame[effective_column] = effective
    if overwrite_source_column:
        frame[raw_column] = effective
    return frame


def field_axis_display_name(axis: str | None) -> str:
    """Return a user-facing label for field axes under the Bz-first convention."""

    normalized = str(axis or "").strip()
    mapping = {
        "bz_mT": "Bz-effective",
        BZ_EFFECTIVE_COLUMN: "Bz-effective",
        BZ_RAW_COLUMN: "Bz-raw",
        "bmag_mT": "|B|",
        "bproj_mT": "Bproj",
        "bx_mT": "Bx",
        "by_mT": "By",
    }
    return mapping.get(normalized, normalized or "Field")


def rms(series: pd.Series) -> float:
    """Return root mean square of a numeric series."""

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(numeric.to_numpy(dtype=float)))))


def canonicalize_waveform_type(value: object) -> str | None:
    """Normalize waveform labels from UI, metadata, and filenames."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    normalized = normalize_name(str(value))
    if not normalized:
        return None

    for canonical, aliases in WAVEFORM_ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def infer_waveform_from_text(*values: object) -> str | None:
    """Infer waveform type from metadata or file name text."""

    structured = infer_conditions_from_filename(*values)
    if structured.get("waveform_type") is not None:
        return str(structured["waveform_type"])

    for value in values:
        waveform_type = canonicalize_waveform_type(value)
        if waveform_type is not None:
            return waveform_type

    haystack = " ".join(str(value) for value in values if value is not None).lower()
    if "triangle" in haystack or "삼각" in haystack or re.search(r"\btri\b", haystack):
        return "triangle"
    if "sine" in haystack or "sin" in haystack or "사인" in haystack:
        return "sine"
    return None


def infer_frequency_from_text(*values: object) -> float | None:
    """Infer frequency in Hz from text such as `0.5주파수` or `10Hz`."""

    for value in values:
        if value is None:
            continue
        text = str(value)
        direct = re.search(r"([-+]?\d+(?:\.\d+)?)\s*hz", text, flags=re.IGNORECASE)
        if direct is not None:
            return float(direct.group(1))
        freq_like = re.search(r"([-+]?\d+(?:\.\d+)?)\s*주파수", text)
        if freq_like is not None:
            return float(freq_like.group(1))
    structured = infer_conditions_from_filename(*values)
    if structured.get("freq_hz") is not None:
        return float(structured["freq_hz"])
    return None


def infer_current_from_text(*values: object) -> float | None:
    """Infer current target in ampere from text."""

    structured = infer_conditions_from_filename(*values)
    if structured.get("current_target_a") is not None:
        return float(structured["current_target_a"])

    for value in values:
        if value is None:
            continue
        text = str(value)
        match = re.search(r"(?<![0-9a-z])([-+]?\d+(?:\.\d+)?)\s*a(?:pp|pk)?\b", text, flags=re.IGNORECASE)
        if match is not None:
            return float(match.group(1))
    return None


def infer_conditions_from_filename(*values: object) -> dict[str, float | str | None]:
    """Infer waveform, frequency, and current from structured file names."""

    for value in values:
        if value is None:
            continue
        text = str(value)
        candidates = [text]
        try:
            candidates.append(Path(text).stem)
        except OSError:
            pass

        for candidate in candidates:
            match = STRUCTURED_FILENAME_PATTERN.search(candidate)
            if match is None:
                transient_match = TRANSIENT_FILENAME_PATTERN.search(candidate)
                if transient_match is None:
                    continue
                return {
                    "waveform_type": None,
                    "freq_hz": float(transient_match.group("freq").replace("p", ".")),
                    "cycle_count": float(transient_match.group("cycle").replace("p", ".")),
                    "current_target_a": float(transient_match.group("current").replace("p", ".")),
                    "current_target_mode": "pp" if "pp" in transient_match.group("mode").lower() else None,
                }
            waveform_token = match.group("waveform").lower()
            waveform_type = "triangle" if waveform_token in {"triangle", "tri"} else "sine"
            return {
                "waveform_type": waveform_type,
                "freq_hz": float(match.group("freq").replace("p", ".")),
                "cycle_count": None,
                "current_target_a": float(match.group("current").replace("p", ".")),
                "current_target_mode": None,
            }

    return {
        "waveform_type": None,
        "freq_hz": None,
        "cycle_count": None,
        "current_target_a": None,
        "current_target_mode": None,
    }


def make_test_id(
    source_file: str,
    sheet_name: str,
    waveform_type: str | None,
    freq_hz: float | None,
    current_pp_target_a: float | None,
) -> str:
    """Build a stable test identifier for comparison tables."""

    stem = Path(source_file).stem
    parts = [stem, sheet_name]
    if waveform_type:
        parts.append(waveform_type)
    if freq_hz is not None:
        parts.append(f"{freq_hz:g}Hz")
    if current_pp_target_a is not None:
        parts.append(f"{current_pp_target_a:g}App")
    return "__".join(part for part in parts if part and part != "main")


def combine_temperature_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Average available temperature columns row-wise."""

    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[available].mean(axis=1, skipna=True)


def reconstruct_signed_current_channels(
    frame: pd.DataFrame,
    custom_current_alpha: float = 1.0,
    custom_current_beta: float = 1.0,
) -> dict[str, object]:
    """Add signed-current helper columns for datasets that store current magnitude only."""

    reference_channel = _select_polarity_reference(frame)
    sign_series = _build_polarity_sign(frame, reference_channel) if reference_channel else None
    reconstructed_columns: list[str] = []
    offsets: dict[str, float] = {}

    for channel in ("coil1_current_a", "coil2_current_a"):
        signed_column = channel.replace("_a", "_signed_a")
        if channel not in frame.columns:
            frame[signed_column] = np.nan
            continue

        numeric = pd.to_numeric(frame[channel], errors="coerce")
        if sign_series is None or not _looks_unsigned_current(numeric):
            frame[signed_column] = numeric
            continue

        offset = _estimate_magnitude_baseline(numeric, frame[reference_channel])
        signed = (numeric - offset).clip(lower=0.0) * sign_series
        frame[signed_column] = signed
        reconstructed_columns.append(channel)
        offsets[signed_column] = offset

    signed_columns = [column for column in ("coil1_current_signed_a", "coil2_current_signed_a") if column in frame.columns]
    if signed_columns:
        frame["i_sum_signed"] = frame[signed_columns].sum(axis=1, min_count=1)
        if "coil1_current_signed_a" in frame.columns and "coil2_current_signed_a" in frame.columns:
            frame["i_diff_signed"] = frame["coil1_current_signed_a"] - frame["coil2_current_signed_a"]
            frame["i_custom_signed"] = (
                custom_current_alpha * frame["coil1_current_signed_a"]
                + custom_current_beta * frame["coil2_current_signed_a"]
            )
        else:
            frame["i_diff_signed"] = np.nan
            frame["i_custom_signed"] = np.nan
    else:
        frame["i_sum_signed"] = np.nan
        frame["i_diff_signed"] = np.nan
        frame["i_custom_signed"] = np.nan

    return {
        "reference_channel": reference_channel,
        "reconstructed_columns": reconstructed_columns,
        "offsets": offsets,
    }


def compute_projection(
    bx: pd.Series,
    by: pd.Series,
    bz: pd.Series,
    vector: tuple[float, float, float],
) -> pd.Series:
    """Project the magnetic field onto a user-selected direction vector."""

    vx, vy, vz = vector
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm == 0:
        return pd.Series(np.nan, index=bx.index, dtype=float)
    return (bx * vx + by * vy + bz * vz) / norm


def choose_best_match(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    """Return the strongest alias match from a set of columns."""

    alias_map = {normalize_name(alias): alias for alias in aliases}
    scored: list[tuple[int, int, str]] = []

    for index, column in enumerate(columns):
        normalized_column = normalize_name(column)
        if normalized_column in alias_map:
            scored.append((3, -index, column))
            continue
        for alias_normalized in alias_map:
            if alias_normalized and alias_normalized in normalized_column:
                scored.append((2, -index, column))
                break
            if normalized_column and normalized_column in alias_normalized:
                scored.append((1, -index, column))
                break

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][2]


def flatten_messages(values: Iterable[str]) -> str:
    """Collapse warning/log lists to a stable single string."""

    unique_values = [value for value in dict.fromkeys(value.strip() for value in values if value)]
    return " | ".join(unique_values)


def column_stats(series: pd.Series) -> dict[str, float]:
    """Compute common signal statistics for a numeric series."""

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "peak": float("nan"),
            "valley": float("nan"),
            "peak_to_peak": float("nan"),
        }

    peak = float(numeric.max())
    valley = float(numeric.min())
    return {
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "rms": rms(numeric),
        "peak": peak,
        "valley": valley,
        "peak_to_peak": peak - valley,
    }


def _select_polarity_reference(frame: pd.DataFrame) -> str | None:
    for channel in ("daq_input_v", "bz_mT", "bproj_mT", "by_mT", "bx_mT"):
        if channel not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[channel], errors="coerce").dropna()
        if len(numeric) < 8:
            continue
        centered = numeric - float(numeric.median())
        if centered.quantile(0.05) < 0 < centered.quantile(0.95):
            return channel
    return None


def _build_polarity_sign(frame: pd.DataFrame, reference_channel: str | None) -> pd.Series | None:
    if reference_channel is None or reference_channel not in frame.columns:
        return None

    numeric = pd.to_numeric(frame[reference_channel], errors="coerce")
    centered = numeric - float(numeric.median())
    sign = np.sign(centered).replace(0, np.nan)
    return sign.ffill().bfill().fillna(1.0)


def _looks_unsigned_current(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 8:
        return False

    negative_ratio = float((numeric < 0).mean())
    peak = float(numeric.abs().max())
    if not np.isfinite(peak) or peak <= 0:
        return False

    return negative_ratio <= 0.02 and float(numeric.min()) >= -0.1 * peak


def _estimate_magnitude_baseline(series: pd.Series, reference_series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    reference = pd.to_numeric(reference_series, errors="coerce")
    valid = numeric.notna() & reference.notna()
    if valid.sum() < 8:
        baseline = numeric.quantile(0.01)
        return float(baseline) if pd.notna(baseline) else 0.0

    centered_reference = (reference.loc[valid] - float(reference.loc[valid].median())).abs()
    threshold = float(centered_reference.quantile(0.1))
    near_zero_mask = valid.copy()
    near_zero_mask.loc[valid] = centered_reference <= threshold

    baseline_series = numeric.loc[near_zero_mask]
    if baseline_series.dropna().empty:
        baseline_series = numeric
    baseline = baseline_series.quantile(0.1)
    if pd.isna(baseline):
        return 0.0
    return float(max(baseline, 0.0))
