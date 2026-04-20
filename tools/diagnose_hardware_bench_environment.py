from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


POLICY_DIR = Path("artifacts/policy_eval")
OUTPUT_JSON = POLICY_DIR / "hardware_environment_diagnostics.json"
OUTPUT_MD = POLICY_DIR / "hardware_environment_diagnostics.md"


def _run_powershell(command: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _run_python_snippet(code: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        [r"D:\programs\Codex\.venv\Scripts\python.exe", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()

    _, usb_6451_out, _ = _run_powershell(
        "Get-PnpDevice | Where-Object { $_.FriendlyName -eq 'USB-6451' } "
        "| Select-Object Status,Present,Class,FriendlyName,InstanceId,Problem,ConfigManagerErrorCode,Service "
        "| ConvertTo-Json -Depth 3"
    )
    usb_6451 = json.loads(usb_6451_out) if usb_6451_out else None

    _, lcr_out, _ = _run_powershell(
        "Get-PnpDevice | Where-Object { $_.FriendlyName -like '*LCR Meter*' } "
        "| Select-Object Status,Present,Class,FriendlyName,InstanceId,Problem,ConfigManagerErrorCode,Service "
        "| ConvertTo-Json -Depth 3"
    )
    lcr_meter = json.loads(lcr_out) if lcr_out else None

    _, ni_service_out, _ = _run_powershell(
        "Get-Service | Where-Object { $_.DisplayName -match 'National Instruments|NI ' -or $_.Name -match '^ni' } "
        "| Select-Object Status,Name,DisplayName | ConvertTo-Json -Depth 3"
    )
    ni_services = json.loads(ni_service_out) if ni_service_out else []

    _, comports_out, _ = _run_python_snippet(
        "import json, serial.tools.list_ports; "
        "ports=[{'device':p.device,'description':p.description,'hwid':p.hwid} for p in serial.tools.list_ports.comports()]; "
        "print(json.dumps(ports, ensure_ascii=False))"
    )
    pyserial_ports = json.loads(comports_out) if comports_out else []

    _, visa_out, _ = _run_python_snippet(
        "import json, pyvisa; "
        "rm=pyvisa.ResourceManager(); "
        "print(json.dumps({'resources':rm.list_resources()}, ensure_ascii=False))"
    )
    visa_resources = json.loads(visa_out) if visa_out else {"resources": []}

    _, nidaq_out, _ = _run_python_snippet(
        "import json; from nidaqmx.system import System; "
        "sys=System.local(); "
        "print(json.dumps({'driver_version':{'major':sys.driver_version.major_version,'minor':sys.driver_version.minor_version,'update':sys.driver_version.update_version},"
        "'devices':[{'name':d.name,'product_type':getattr(d,'product_type',None)} for d in sys.devices]}, ensure_ascii=False))"
    )
    nidaq_info = json.loads(nidaq_out) if nidaq_out else {}

    _, repo_scan_out, _ = _run_powershell(
        "Get-ChildItem -Recurse -File | Select-String -Pattern 'nidaqmx|DAQmx|pyvisa|serial\\.Serial|ResourceManager|COM5|USB-6451|LCR Meter' "
        "| Select-Object Path,LineNumber,Line | ConvertTo-Json -Depth 3"
    )
    repo_hits = json.loads(repo_scan_out) if repo_scan_out else []
    repo_has_bench_runner = bool(repo_hits)

    blockers: list[str] = []
    if not usb_6451:
        blockers.append("USB-6451가 PnP 조회 결과에 없습니다.")
    elif not usb_6451.get("Present", False):
        blockers.append("USB-6451가 Windows에서 phantom device로 남아 있고 Present=False 상태입니다.")

    if not nidaq_info.get("devices"):
        blockers.append("NI-DAQmx Python API에서 인식되는 실제 DAQ device가 0개입니다.")

    if not lcr_meter:
        blockers.append("LCR Meter COM 포트가 현재 조회 결과에 없습니다.")
    elif not lcr_meter.get("Present", False):
        blockers.append("LCR Meter Virtual COM Port(COM5)가 Windows에서 phantom device이며 Present=False 상태입니다.")

    port_names = {item.get("device") for item in pyserial_ports}
    if "COM5" not in port_names:
        blockers.append("pyserial 기준으로 COM5가 현재 열거되지 않아 LCR meter 자동 bench를 시작할 수 없습니다.")

    resources = set(visa_resources.get("resources", []))
    if not any("COM5" in item or "USB" in item or "GPIB" in item for item in resources):
        blockers.append("VISA 자원 목록에 bench 계측기나 COM5가 보이지 않습니다.")

    if not repo_has_bench_runner:
        blockers.append("현재 워크스페이스에는 export 파일을 장비에 인가하고 계측까지 수행하는 bench runner 코드가 없습니다.")

    payload = {
        "generated_at_utc": generated_at,
        "usb_6451": usb_6451,
        "lcr_meter": lcr_meter,
        "ni_services_running": ni_services,
        "pyserial_ports": pyserial_ports,
        "visa_resources": visa_resources.get("resources", []),
        "nidaqmx": nidaq_info,
        "repo_has_bench_runner_signals": repo_has_bench_runner,
        "repo_hits": repo_hits if isinstance(repo_hits, list) else [repo_hits],
        "blockers": blockers,
        "summary": {
            "can_execute_bench_here": False if blockers else True,
            "root_cause": "physical_device_absent_or_not_present" if blockers else "none",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Hardware Environment Diagnostics",
        "",
        f"- generated_at_utc: `{generated_at}`",
        "",
        "## Summary",
        "",
        f"- can_execute_bench_here: `{payload['summary']['can_execute_bench_here']}`",
        f"- root_cause: `{payload['summary']['root_cause']}`",
        "",
        "## USB-6451",
        "",
        f"- status: `{(usb_6451 or {}).get('Status')}`",
        f"- present: `{(usb_6451 or {}).get('Present')}`",
        f"- problem: `{(usb_6451 or {}).get('Problem')}`",
        f"- service: `{(usb_6451 or {}).get('Service')}`",
        "",
        "## LCR Meter COM5",
        "",
        f"- status: `{(lcr_meter or {}).get('Status')}`",
        f"- present: `{(lcr_meter or {}).get('Present')}`",
        f"- problem: `{(lcr_meter or {}).get('Problem')}`",
        f"- service: `{(lcr_meter or {}).get('Service')}`",
        "",
        "## NI-DAQmx Python",
        "",
        f"- driver_version: `{nidaq_info.get('driver_version')}`",
        f"- detected_devices: `{nidaq_info.get('devices', [])}`",
        "",
        "## pyserial / VISA",
        "",
        f"- pyserial_ports: `{pyserial_ports}`",
        f"- visa_resources: `{visa_resources.get('resources', [])}`",
        "",
        "## Bench Runner Code",
        "",
        f"- repo_has_bench_runner_signals: `{repo_has_bench_runner}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in blockers:
        lines.append(f"- {blocker}")
    if not blockers:
        lines.append("- no blocking issue detected")

    lines.extend(
        [
            "",
            "## Immediate Next Steps",
            "",
            "- USB-6451와 LCR meter를 실제로 다시 연결하고, Windows Device Manager/NI MAX에서 Present 상태인지 확인",
            "- COM5가 pyserial에서 다시 보이면 LCR meter handshake를 별도로 점검",
            "- USB-6451가 NI-DAQmx `System.devices`에 나타나면 그 다음에야 bench runner 자동화 또는 수동 smoke test를 실행 가능",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
