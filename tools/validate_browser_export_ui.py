from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By


APP_PORT = 8526
OUTPUT_JSON = Path("artifacts/policy_eval/browser_export_validation.json")
DOWNLOAD_DIR = Path("artifacts/policy_eval/browser_downloads")


def _visible_button_texts(driver: webdriver.Chrome) -> list[str]:
    return [button.text.strip() for button in driver.find_elements(By.TAG_NAME, "button") if (button.text or "").strip()]


def _wait_for_downloads(download_dir: Path, expected_count: int, timeout_s: float = 45.0) -> list[Path]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        partials = list(download_dir.glob("*.crdownload"))
        files = [path for path in download_dir.iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".txt"}]
        if not partials and len(files) >= expected_count:
            return sorted(files)
        time.sleep(1.0)
    return sorted(path for path in download_dir.iterdir() if path.is_file())


def _click_download_button(driver: webdriver.Chrome, substring: str) -> str | None:
    for button in driver.find_elements(By.TAG_NAME, "button"):
        text = (button.text or "").strip()
        if substring in text:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", button)
            ActionChains(driver).move_to_element(button).click(button).perform()
            return text
    return None


def _inspect_download(path: Path) -> dict[str, object]:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in content.splitlines() if line.strip()]
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "line_count": len(lines),
        "header": lines[0] if lines else "",
        "has_comma_header": bool(lines and "," in lines[0]),
    }


def main() -> int:
    root = Path.cwd()
    python_exe = root.parent / ".venv" / "Scripts" / "python.exe"
    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            str(python_exe),
            "-m",
            "streamlit",
            "run",
            "app_field_analysis_quick.py",
            "--server.headless",
            "true",
            "--server.port",
            str(APP_PORT),
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    driver: webdriver.Chrome | None = None
    try:
        time.sleep(12)
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1800,2600")
        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": str(DOWNLOAD_DIR.resolve()),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
            },
        )
        driver = webdriver.Chrome(options=options)
        driver.get(f"http://127.0.0.1:{APP_PORT}")
        time.sleep(220)

        before_buttons = _visible_button_texts(driver)
        field_button = next(
            button for button in driver.find_elements(By.TAG_NAME, "button") if "bz_mT" in (button.text or "")
        )
        field_button.click()
        time.sleep(35)

        body_text = driver.find_element(By.TAG_NAME, "body").text
        after_buttons = _visible_button_texts(driver)
        export_buttons = [
            text
            for text in after_buttons
            if "다운로드" in text or "download" in text.lower() or "?ㅼ슫濡쒕뱶" in text
        ]

        clicked_buttons: list[str] = []
        for label in ["제어 LUT CSV", "보정 전압 파형 CSV"]:
            clicked = _click_download_button(driver, label)
            if clicked:
                clicked_buttons.append(clicked)
                time.sleep(2.0)

        downloaded_files = _wait_for_downloads(DOWNLOAD_DIR, expected_count=len(clicked_buttons))
        inspected_downloads = [_inspect_download(path) for path in downloaded_files]

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "validation_method": "selenium + headless chrome",
            "scope": "default exact field path export rendering + selected file downloads",
            "limitations": [
                "Current-target browser rerender was not stabilized in this headless path; current-target state validation remains covered by AppTest artifact.",
                "Validation confirms rendered export buttons and downloaded file presence/basic content, not full numerical golden-file comparison.",
            ],
            "before_buttons": before_buttons,
            "after_buttons": after_buttons,
            "export_buttons": export_buttons,
            "clicked_download_buttons": clicked_buttons,
            "download_dir": str(DOWNLOAD_DIR.resolve()),
            "downloaded_files": inspected_downloads,
            "found_completion_message": "완료" in body_text or "?꾨즺" in body_text,
            "found_export_section": (
                "보정 LUT 지원점" in body_text
                or "제어 전달" in body_text
                or "蹂댁젙 LUT" in body_text
            ),
            "found_harmonic_transfer_download": any("Harmonic Transfer LUT" in text for text in export_buttons),
            "download_content_validation_passed": bool(
                inspected_downloads
                and all(
                    item["size_bytes"] > 0 and item["line_count"] >= 2 and item["has_comma_header"]
                    for item in inspected_downloads
                )
            ),
        }
        OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {OUTPUT_JSON}")
        return 0
    finally:
        if driver is not None:
            driver.quit()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
