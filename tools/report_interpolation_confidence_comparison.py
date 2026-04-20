from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


POLICY_DIR = Path("artifacts/policy_eval")
OUTPUT_JSON = POLICY_DIR / "interpolation_confidence_comparison.json"
OUTPUT_MD = POLICY_DIR / "interpolation_confidence_comparison.md"


def _read_json(name: str) -> dict:
    return json.loads((POLICY_DIR / name).read_text(encoding="utf-8"))


def _row(label: str, payload: dict) -> dict[str, object]:
    confusion = payload.get("confusion_counts", {})
    return {
        "label": label,
        "policy_version": payload.get("policy_version"),
        "case_count": int(payload.get("case_count", 0) or 0),
        "auto_count": int(payload.get("auto_count", 0) or 0),
        "false_auto_count": int(payload.get("false_auto_count", 0) or 0),
        "false_block_count": int(confusion.get("false_block", 0) or 0),
        "correct_auto_count": int(confusion.get("correct_auto", 0) or 0),
        "mean_realized_shape_corr": float(payload.get("mean_realized_shape_corr", float("nan"))),
        "mean_realized_nrmse": float(payload.get("mean_realized_nrmse", float("nan"))),
        "mean_predicted_error_band": float(payload.get("mean_predicted_error_band", float("nan"))),
    }


def main() -> int:
    v2 = _read_json("policy_eval_v2_continuous_corpus_l1fo.json")
    v3_candidate = _read_json("policy_eval_v3_candidate_p95_continuous_corpus_l1fo.json")
    v3_geom = _read_json("policy_eval_v3_geom_p95_continuous_corpus_l1fo.json")
    rows = [
        _row("v2_operational", v2),
        _row("v3_candidate_p95", v3_candidate),
        _row("v3_geom_p95", v3_geom),
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "conclusion": {
            "operational_policy": "v2 유지",
            "interpolated_auto": "계속 닫힘",
            "geometry_redesign_effect": "false_auto는 제거했지만 auto 승격 근거는 아직 부족",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Interpolation Confidence Comparison",
        "",
        "| label | policy_version | case_count | auto_count | false_auto | false_block | correct_auto | mean_shape_corr | mean_nrmse | mean_predicted_error_band |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["policy_version"]),
                    str(row["case_count"]),
                    str(row["auto_count"]),
                    str(row["false_auto_count"]),
                    str(row["false_block_count"]),
                    str(row["correct_auto_count"]),
                    f"{float(row['mean_realized_shape_corr']):.6f}",
                    f"{float(row['mean_realized_nrmse']):.6f}",
                    f"{float(row['mean_predicted_error_band']):.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- 운영 정책은 계속 `v2`입니다.",
            "- `v3_candidate_p95`는 false auto가 많아 운영에 부적합합니다.",
            "- `v3_geom_p95`는 false auto를 제거했지만, interpolated auto를 다시 열 근거는 만들지 못했습니다.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
