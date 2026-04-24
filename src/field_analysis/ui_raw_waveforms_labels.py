from __future__ import annotations

import re
from pathlib import PurePath


FIXED_DAQ_OUTPUT_LABEL = "±5V"
FIXED_GAIN_LABEL = "Gain 100%"

_NEW_DATASET_FILENAME_PATTERN = re.compile(
    r"^(?P<source_type>continuous|finite)_(?P<waveform>sine|sin|triangle|tri)_"
    r"(?P<freq>\d+(?:[.p]\d+)?)hz(?:_(?P<cycle>\d+(?:[.p]\d+)?)cycle)?$",
    re.IGNORECASE,
)
_OPAQUE_PREFIX_PATTERN = re.compile(r"^[0-9a-f]{12,}_", re.IGNORECASE)


def infer_new_dataset_filename_metadata(file_name: object) -> dict[str, float | str | None]:
    leaf_name = str(file_name or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = PurePath(_OPAQUE_PREFIX_PATTERN.sub("", leaf_name)).stem
    match = _NEW_DATASET_FILENAME_PATTERN.match(stem)
    if match is None:
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
    }
