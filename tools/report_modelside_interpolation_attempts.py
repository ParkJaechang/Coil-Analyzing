from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


POLICY_DIR = Path("artifacts/policy_eval")
OUTPUT_JSON = POLICY_DIR / "modelside_interpolation_attempts.json"
OUTPUT_MD = POLICY_DIR / "modelside_interpolation_attempts.md"


def _read_json(name: str) -> dict:
    return json.loads((POLICY_DIR / name).read_text(encoding="utf-8"))


def _row(label: str, payload: dict) -> dict[str, object]:
    confusion = payload.get("confusion_counts", {})
    return {
        "label": label,
        "policy_version": payload.get("policy_version"),
        "case_count": int(payload.get("case_count", 0) or 0),
        "auto_count": int(payload.get("auto_count", 0) or 0),
        "false_auto_count": int(confusion.get("false_auto", 0) or 0),
        "false_block_count": int(confusion.get("false_block", 0) or 0),
        "mean_shape_corr": float(payload.get("mean_realized_shape_corr", float("nan"))),
        "mean_nrmse": float(payload.get("mean_realized_nrmse", float("nan"))),
        "mean_predicted_error_band": float(payload.get("mean_predicted_error_band", float("nan"))),
    }


def main() -> int:
    baseline = _read_json("policy_eval_v2_continuous_corpus_l1fo.json")
    localbracket = _read_json("policy_eval_v2_modelside_localbracket_continuous_corpus_l1fo.json")
    phaseanchor = _read_json("policy_eval_v2_modelside_phaseanchor_continuous_corpus_l1fo.json")

    rows = [
        _row("baseline_v2", baseline),
        _row("localbracket_attempt", localbracket),
        _row("phaseanchor_attempt", phaseanchor),
    ]

    baseline_nrmse = rows[0]["mean_nrmse"]
    local_nrmse = rows[1]["mean_nrmse"]
    phaseanchor_nrmse = rows[2]["mean_nrmse"]
    if phaseanchor_nrmse < baseline_nrmse - 1e-3:
        conclusion = "phase anchor 시도가 baseline 대비 의미 있는 개선을 만들었습니다."
    elif local_nrmse < baseline_nrmse - 1e-3:
        conclusion = "local bracket 시도가 baseline 대비 의미 있는 개선을 만들었습니다."
    else:
        conclusion = "이번 model-side interpolation 2회 시도는 baseline 대비 유의미한 quality 개선을 만들지 못했습니다."

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "conclusion": conclusion,
        "operational_decision": "interpolated auto remains closed",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Model-side Interpolation Attempts",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['label']}",
                "",
                f"- policy_version: `{row['policy_version']}`",
                f"- case_count: `{row['case_count']}`",
                f"- auto_count: `{row['auto_count']}`",
                f"- false_auto_count: `{row['false_auto_count']}`",
                f"- false_block_count: `{row['false_block_count']}`",
                f"- mean_shape_corr: `{row['mean_shape_corr']:.6f}`",
                f"- mean_nrmse: `{row['mean_nrmse']:.6f}`",
                f"- mean_predicted_error_band: `{row['mean_predicted_error_band']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            f"- {conclusion}",
            "- exact path regression: `none observed in exact export validation`",
            "- operational decision: `interpolated auto remains closed`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
