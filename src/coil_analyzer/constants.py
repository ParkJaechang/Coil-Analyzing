"""Application-wide constants."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "Coil Large-Signal Analyzer"
APP_DIRNAME = ".coil_analyzer"
UPLOAD_DIRNAME = "uploads"
EXPORT_DIRNAME = "exports"
CONFIG_DIRNAME = "configs"
MANIFEST_FILENAME = "manifest.json"
MAPPING_LIBRARY_FILENAME = "mapping_library.json"
REFERENCE_FILE_CANDIDATES = (
    "all_bands_full.xlsx",
    "7224-Datasheet-05-06-24 (1).pdf",
    "7224-7226_OperatorManual-1.pdf",
    "코스모크 전자석_Silicon steel (25.10.28).pdf",
)
DEFAULT_REQUEST_FREQUENCIES_HZ = [0.25, 0.5, 1.0, 1.25, 2.0, 3.0, 4.0, 5.0]
DEFAULT_TARGET_IPP_A = 20.0
TEST_STATUSES = ("not tested", "data loaded", "analyzed", "flagged")
GAIN_MODES = (20.0, 6.0)
DEFAULT_INPUT_VIN_PK = 9.0
TIME_UNIT_FACTORS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}


def workspace_app_dir(root: Path) -> Path:
    return root / APP_DIRNAME
