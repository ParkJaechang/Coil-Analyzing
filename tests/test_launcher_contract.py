from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = [
    REPO_ROOT / "launch_streamlit_with_free_port_local.cmd",
    REPO_ROOT / "launch_streamlit_with_free_port.cmd",
]


def test_launchers_run_streamlit_headless_and_log_runtime_contract() -> None:
    for launcher in LAUNCHERS:
        source = launcher.read_text(encoding="utf-8")
        assert "--server.headless true" in source
        assert "selected APP_SCRIPT" in source
        assert "APP_PORT" in source
        assert "APP_URL" in source
        assert "BROWSER_AUTO_OPEN" in source
        assert "LAUNCH_STATUS" in source


def test_launchers_open_browser_at_most_once_and_honor_disable_env() -> None:
    for launcher in LAUNCHERS:
        source = launcher.read_text(encoding="utf-8")
        assert 'if /I not "%FIELD_ANALYSIS_OPEN_BROWSER%"=="0"' in source
        assert source.count('start "" "%APP_URL%"') <= 1


def test_launchers_reuse_existing_server_lock() -> None:
    for launcher in LAUNCHERS:
        source = launcher.read_text(encoding="utf-8")
        assert "FIELD_ANALYSIS_APP_LOCK_DIR" in source
        assert "REUSED_EXISTING_SERVER" in source
        assert "_stcore/health" in source
