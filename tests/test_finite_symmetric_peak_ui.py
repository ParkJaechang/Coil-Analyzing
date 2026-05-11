from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_raw_waveforms_normalization_and_symmetric_peak_ui_markers_exist() -> None:
    raw_ui = (REPO_ROOT / "src" / "field_analysis" / "ui_raw_waveforms.py").read_text(encoding="utf-8")
    plot_ui = (REPO_ROOT / "src" / "field_analysis" / "ui_raw_waveforms_plot.py").read_text(encoding="utf-8")

    assert "build_finite_symmetric_peak_review" in raw_ui
    assert "render_finite_symmetric_peak_review" in raw_ui
    assert "Raw vs normalized field" in plot_ui
    assert "Normalization panel" in plot_ui
    assert "normalization enabled" in plot_ui
    assert "source peak" in plot_ui
    assert "scale factor" in plot_ui
    assert "active window start/end" in plot_ui
    assert "raw peak" in plot_ui
    assert "normalized peak" in plot_ui
    assert "active segment 기준 정규화" in plot_ui
    assert "pre/post rest는 scale 계산에서 제외" in plot_ui


def test_finite_symmetric_peak_ui_policy_and_metrics_markers_exist() -> None:
    plot_ui = (REPO_ROOT / "src" / "field_analysis" / "ui_raw_waveforms_plot.py").read_text(encoding="utf-8")

    expected = [
        "Finite symmetric peak review",
        "지원 cycles: 1.0 / 1.5",
        "미지원 cycles: 1.25 / 1.75",
        "phase-delay peak correction disabled",
        "unsupported_cycle",
        "positive_peak_mT",
        "negative_peak_mT",
        "peak_symmetry_error_mT",
        "peak_symmetry_ratio",
        "command_voltage_peak_v",
        "command_voltage_limit_status",
        "normalized physical target",
        "normalized field/predicted field",
        "positive/negative lobe markers",
        "baseline command",
        "symmetric peak command candidate",
        "command delta",
        "residual / lobe error",
        "절대 gain 평가가 아니라 개형/대칭성 검토용입니다.",
        "1.25 / 1.75는 phase delay 때문에 peak amplitude correction 주력 경로에서 제외합니다.",
        "1.0 / 1.5 finite-cycle에 대해 양/음 peak symmetry를 검토합니다.",
        "Raw data는 보존하고 normalized data는 review/modeling용입니다.",
    ]
    missing = [marker for marker in expected if marker not in plot_ui]

    assert not missing


def test_raw_waveforms_new_ui_text_has_no_mojibake() -> None:
    files = [
        REPO_ROOT / "src" / "field_analysis" / "ui_raw_waveforms_plot.py",
        REPO_ROOT / "src" / "field_analysis" / "waveform_review_normalization.py",
    ]
    mojibake_patterns = (
        chr(0xFFFD),
        chr(0xF9E4),
        chr(0xC4D2),
        "?" + chr(0xAFF0) + chr(0xC0AC),
    )

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not any(pattern in text for pattern in mojibake_patterns), path
