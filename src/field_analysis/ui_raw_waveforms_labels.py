from __future__ import annotations

import re
from pathlib import PurePath

from .upload_filename import canonical_upload_filename
from .voltage_policy import COMMAND_VOLTAGE_LIMIT_LABEL


FIXED_DAQ_OUTPUT_LABEL = COMMAND_VOLTAGE_LIMIT_LABEL
FIXED_GAIN_LABEL = "Gain 100%"

_NEW_DATASET_FILENAME_PATTERN = re.compile(
    r"^(?P<source_type>continuous|finite)_(?P<waveform>sine|sin|triangle|tri)_"
    r"(?P<freq>\d+(?:[.p]\d+)?)hz(?:_(?P<cycle>\d+(?:[.p]\d+)?)cycle)?$",
    re.IGNORECASE,
)
_DEFAULT_TRIANGLE_CONTINUOUS_PATTERN = re.compile(
    r"^continuous_(?P<freq>\d+(?:[.p]\d+)?)hz$",
    re.IGNORECASE,
)
_DEFAULT_TRIANGLE_FINITE_PATTERN = re.compile(
    r"^finite_(?P<cycle>\d+(?:[.p]\d+)?)cycle_(?P<freq>\d+(?:[.p]\d+)?)hz$",
    re.IGNORECASE,
)
_OPAQUE_PREFIX_PATTERN = re.compile(r"^[0-9a-f]{12,}_", re.IGNORECASE)


def infer_new_dataset_filename_metadata(file_name: object) -> dict[str, float | str | None]:
    leaf_name = canonical_upload_filename(str(file_name or "").replace("\\", "/").rsplit("/", 1)[-1])
    stem = PurePath(_OPAQUE_PREFIX_PATTERN.sub("", leaf_name)).stem
    match = _NEW_DATASET_FILENAME_PATTERN.match(stem)
    if match is None:
        continuous_match = _DEFAULT_TRIANGLE_CONTINUOUS_PATTERN.match(stem)
        if continuous_match is not None:
            return {
                "source_type": "continuous",
                "waveform_type": "triangle",
                "freq_hz": float(continuous_match.group("freq").replace("p", ".")),
                "cycle_count": None,
                "waveform_source": "default_triangle_filename",
            }
        finite_match = _DEFAULT_TRIANGLE_FINITE_PATTERN.match(stem)
        if finite_match is not None:
            return {
                "source_type": "finite-cycle",
                "waveform_type": "triangle",
                "freq_hz": float(finite_match.group("freq").replace("p", ".")),
                "cycle_count": float(finite_match.group("cycle").replace("p", ".")),
                "waveform_source": "default_triangle_filename",
            }
        return {"source_type": None, "waveform_type": None, "freq_hz": None, "cycle_count": None}

    source_type = "finite-cycle" if match.group("source_type").lower() == "finite" else "continuous"
    waveform_token = match.group("waveform").lower()
    waveform_type = "triangle" if waveform_token in {"triangle", "tri"} else "sine"
    cycle_text = match.group("cycle")
    return {
        "source_type": source_type,
        "waveform_type": waveform_type,
        "freq_hz": float(match.group("freq").replace("p", ".")),
        "cycle_count": float(cycle_text.replace("p", ".")) if cycle_text is not None else None,
        "waveform_source": "filename",
    }
