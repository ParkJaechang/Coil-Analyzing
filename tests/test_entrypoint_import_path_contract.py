from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    (
        "quick",
        REPO_ROOT / "app_field_analysis_quick.py",
        "run_quick_lut_app",
    ),
    (
        "latest",
        REPO_ROOT / "app_field_analysis_latest.py",
        "run_app",
    ),
]


def _write_stub_app_ui(package_dir: Path, *, marker: str) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "app_ui.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def run_quick_lut_app() -> None:",
                f"    print('MARKER={marker}')",
                "    print(f'MODULE_PATH={Path(__file__).resolve()}')",
                "",
                "def run_app() -> None:",
                f"    print('MARKER={marker}')",
                "    print(f'MODULE_PATH={Path(__file__).resolve()}')",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_entrypoint_in_shadow_layout(
    tmp_path: Path,
    *,
    entrypoint_path: Path,
) -> subprocess.CompletedProcess[str]:
    temp_repo_root = tmp_path / "repo"
    temp_repo_root.mkdir(parents=True, exist_ok=True)
    temp_entrypoint = temp_repo_root / entrypoint_path.name
    temp_entrypoint.write_text(entrypoint_path.read_text(encoding="utf-8"), encoding="utf-8")

    repo_local_package = temp_repo_root / "src" / "field_analysis"
    parent_shadow_package = temp_repo_root.parent / "src" / "field_analysis"
    _write_stub_app_ui(repo_local_package, marker="repo-local")
    _write_stub_app_ui(parent_shadow_package, marker="parent-shadow")

    return subprocess.run(
        [sys.executable, "-I", str(temp_entrypoint)],
        cwd=str(temp_repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def test_entrypoint_sources_use_same_repo_first_guardrail_policy() -> None:
    quick_source = (REPO_ROOT / "app_field_analysis_quick.py").read_text(encoding="utf-8")
    latest_source = (REPO_ROOT / "app_field_analysis_latest.py").read_text(encoding="utf-8")

    for source in (quick_source, latest_source):
        assert 'PROJECT_ROOT / "src"' in source
        assert 'PROJECT_ROOT.parent / "src"' in source
        assert "def _prepend_sys_path" in source
        assert "def _install_src_path_guardrails" in source
        assert "reversed(SRC_CANDIDATES)" in source
        assert 'field_analysis.app_ui' in source
        assert 'field_analysis.app_ui_snapshot' in source


def test_quick_and_latest_entrypoints_have_matching_guardrail_helpers() -> None:
    quick_source = (REPO_ROOT / "app_field_analysis_quick.py").read_text(encoding="utf-8")
    latest_source = (REPO_ROOT / "app_field_analysis_latest.py").read_text(encoding="utf-8")

    quick_helper = quick_source.split("LAST_ERROR:", maxsplit=1)[0]
    latest_helper = latest_source.split("LAST_ERROR:", maxsplit=1)[0]

    normalized_quick = quick_helper.replace("run_quick_lut_app", "RUN_FUNC").replace(
        "app_field_analysis_quick.py",
        "ENTRYPOINT.py",
    )
    normalized_latest = latest_helper.replace("run_app", "RUN_FUNC").replace(
        "app_field_analysis_latest.py",
        "ENTRYPOINT.py",
    )

    assert normalized_quick == normalized_latest


def test_quick_entrypoint_prefers_repo_local_src_over_parent_shadow(tmp_path: Path) -> None:
    result = _run_entrypoint_in_shadow_layout(
        tmp_path,
        entrypoint_path=REPO_ROOT / "app_field_analysis_quick.py",
    )

    assert result.returncode == 0, result.stderr
    assert "MARKER=repo-local" in result.stdout
    assert "parent-shadow" not in result.stdout
    assert str((tmp_path / "repo" / "src" / "field_analysis" / "app_ui.py").resolve()) in result.stdout


def test_latest_entrypoint_prefers_repo_local_src_over_parent_shadow(tmp_path: Path) -> None:
    result = _run_entrypoint_in_shadow_layout(
        tmp_path,
        entrypoint_path=REPO_ROOT / "app_field_analysis_latest.py",
    )

    assert result.returncode == 0, result.stderr
    assert "MARKER=repo-local" in result.stdout
    assert "parent-shadow" not in result.stdout
    assert str((tmp_path / "repo" / "src" / "field_analysis" / "app_ui.py").resolve()) in result.stdout

