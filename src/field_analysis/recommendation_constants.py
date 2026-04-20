from __future__ import annotations

from pathlib import Path


OFFICIAL_OPERATION_MAX_FREQ_HZ = 5.0
REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOT = (
    REPO_ROOT / "Coil Analyzing"
    if (REPO_ROOT / "Coil Analyzing" / "tools" / "generate_bz_first_artifacts.py").exists()
    else REPO_ROOT
)
BZ_FIRST_ARTIFACT_DIR = PRODUCT_ROOT / "artifacts" / "bz_first_exact_matrix"
EXACT_MATRIX_ARTIFACT_PATH = BZ_FIRST_ARTIFACT_DIR / "exact_matrix_final.json"
FINITE_PROVISIONAL_RECIPE_CANDIDATES: tuple[dict[str, float | str], ...] = (
    {
        "waveform": "sine",
        "freq_hz": 1.0,
        "cycles": 1.0,
        "target_level_pp": 20.0,
        "source_level_pp": 10.0,
    },
)
