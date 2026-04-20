# Hardware Environment Diagnostics

- generated_at_utc: `2026-04-14T05:22:32.638221+00:00`

## Summary

- can_execute_bench_here: `False`
- root_cause: `physical_device_absent_or_not_present`

## USB-6451

- status: `Unknown`
- present: `False`
- problem: `45`
- service: `ninimbusrkw`

## LCR Meter COM5

- status: `Unknown`
- present: `False`
- problem: `45`
- service: `FTSER2K`

## NI-DAQmx Python

- driver_version: `{'major': 25, 'minor': 8, 'update': 1}`
- detected_devices: `[]`

## pyserial / VISA

- pyserial_ports: `[{'device': 'COM3', 'description': 'ǥ�� Bluetooth���� ���� ��ũ(COM3)', 'hwid': 'BTHENUM\\{00001101-0000-1000-8000-00805F9B34FB}_VID&0001009E_PID&4066\\8&3A9899E2&0&E458BC5D91E3_C00000000'}, {'device': 'COM4', 'description': 'ǥ�� Bluetooth���� ���� ��ũ(COM4)', 'hwid': 'BTHENUM\\{00001101-0000-1000-8000-00805F9B34FB}_LOCALMFG&0000\\8&3A9899E2&0&000000000000_00000002'}]`
- visa_resources: `['ASRL3::INSTR', 'ASRL4::INSTR']`

## Bench Runner Code

- repo_has_bench_runner_signals: `True`

## Blockers

- USB-6451가 Windows에서는 phantom device로 남아 있고 Present=False 상태입니다.
- NI-DAQmx Python API에서 인식되는 실제 DAQ device가 0개입니다.
- LCR Meter Virtual COM Port(COM5)가 Windows에서 phantom device이며 Present=False 상태입니다.
- pyserial 기준으로 COM5가 현재 열거되지 않아 LCR meter 자동 bench를 시작할 수 없습니다.
- VISA 자원 목록에 bench 계측기나 COM5가 보이지 않습니다.

## Immediate Next Steps

- USB-6451와 LCR meter를 실제로 다시 연결하고, Windows Device Manager/NI MAX에서 Present 상태인지 확인
- COM5가 pyserial에서 다시 보이면 LCR meter handshake를 별도로 점검
- USB-6451가 NI-DAQmx `System.devices`에 나타나면 그 다음에야 bench runner 자동화 또는 수동 smoke test를 실행 가능