from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


POLICY_DIR = Path("artifacts/policy_eval")
BASELINE_JSON = POLICY_DIR / "policy_eval_v2_continuous_corpus_l1fo.json"
BASELINE_CSV = POLICY_DIR / "policy_eval_v2_continuous_corpus_l1fo.csv"
ATTEMPTS = [
    {
        "label": "localbracket",
        "json": POLICY_DIR / "policy_eval_v2_modelside_localbracket_continuous_corpus_l1fo.json",
        "csv": POLICY_DIR / "policy_eval_v2_modelside_localbracket_continuous_corpus_l1fo.csv",
    },
    {
        "label": "phaseanchor",
        "json": POLICY_DIR / "policy_eval_v2_modelside_phaseanchor_continuous_corpus_l1fo.json",
        "csv": POLICY_DIR / "policy_eval_v2_modelside_phaseanchor_continuous_corpus_l1fo.csv",
    },
]
OUTPUT_JSON = POLICY_DIR / "modelside_interpolation_compare.json"
OUTPUT_MD = POLICY_DIR / "modelside_interpolation_compare.md"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "case_count": int(len(frame)),
        "mean_shape_corr": _safe_float(frame["realized_shape_corr"].mean()) if "realized_shape_corr" in frame.columns else float("nan"),
        "mean_nrmse": _safe_float(frame["realized_nrmse"].mean()) if "realized_nrmse" in frame.columns else float("nan"),
        "mean_phase_lag_cycles": _safe_float(frame["realized_phase_lag_cycles"].mean()) if "realized_phase_lag_cycles" in frame.columns else float("nan"),
    }


def _attempt_payload(label: str, baseline_frame: pd.DataFrame, updated_frame: pd.DataFrame, updated_summary: dict) -> dict[str, object]:
    comparison = baseline_frame.merge(
        updated_frame,
        on=["holdout_test_id"],
        suffixes=("_baseline", "_updated"),
        how="inner",
    )
    comparison["shape_corr_delta"] = (
        pd.to_numeric(comparison["realized_shape_corr_updated"], errors="coerce")
        - pd.to_numeric(comparison["realized_shape_corr_baseline"], errors="coerce")
    )
    comparison["nrmse_delta"] = (
        pd.to_numeric(comparison["realized_nrmse_updated"], errors="coerce")
        - pd.to_numeric(comparison["realized_nrmse_baseline"], errors="coerce")
    )
    comparison["phase_lag_delta"] = (
        pd.to_numeric(comparison["realized_phase_lag_cycles_updated"], errors="coerce")
        - pd.to_numeric(comparison["realized_phase_lag_cycles_baseline"], errors="coerce")
    )

    by_waveform_rows = []
    if "waveform_type_baseline" in comparison.columns:
        for waveform, group in comparison.groupby("waveform_type_baseline", dropna=False):
            by_waveform_rows.append(
                {
                    "waveform_type": waveform,
                    "case_count": int(len(group)),
                    "mean_shape_corr_baseline": _safe_float(group["realized_shape_corr_baseline"].mean()),
                    "mean_shape_corr_updated": _safe_float(group["realized_shape_corr_updated"].mean()),
                    "mean_nrmse_baseline": _safe_float(group["realized_nrmse_baseline"].mean()),
                    "mean_nrmse_updated": _safe_float(group["realized_nrmse_updated"].mean()),
                    "mean_shape_corr_delta": _safe_float(group["shape_corr_delta"].mean()),
                    "mean_nrmse_delta": _safe_float(group["nrmse_delta"].mean()),
                    "mean_phase_lag_delta": _safe_float(group["phase_lag_delta"].mean()),
                }
            )

    return {
        "label": label,
        "updated_summary": updated_summary,
        "updated_derived": _frame_summary(updated_frame),
        "delta": {
            "mean_shape_corr_delta": _safe_float(comparison["shape_corr_delta"].mean()),
            "mean_nrmse_delta": _safe_float(comparison["nrmse_delta"].mean()),
            "mean_phase_lag_delta": _safe_float(comparison["phase_lag_delta"].mean()),
        },
        "by_waveform": by_waveform_rows,
    }


def main() -> int:
    baseline_summary = _load_json(BASELINE_JSON)
    baseline_frame = pd.read_csv(BASELINE_CSV)
    baseline_payload = {
        "summary": baseline_summary,
        "derived": _frame_summary(baseline_frame),
    }

    attempts = []
    for item in ATTEMPTS:
        attempts.append(
            _attempt_payload(
                label=item["label"],
                baseline_frame=baseline_frame,
                updated_frame=pd.read_csv(item["csv"]),
                updated_summary=_load_json(item["json"]),
            )
        )

    best_attempt = min(attempts, key=lambda row: row["delta"]["mean_nrmse_delta"])
    if best_attempt["delta"]["mean_nrmse_delta"] < -1e-3:
        conclusion = f"{best_attempt['label']} 시도가 평균 NRMSE 기준으로 baseline 대비 유의미한 개선을 만들었습니다."
    else:
        conclusion = "이번 model-side interpolation 시도들은 평균 NRMSE 기준으로 baseline 대비 유의미한 개선을 만들지 못했습니다."

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline_payload,
        "attempts": attempts,
        "best_attempt_label": best_attempt["label"],
        "conclusion": conclusion,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Model-side Interpolation Compare",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        "",
        "## Baseline",
        "",
        f"- case_count: `{baseline_payload['derived']['case_count']}`",
        f"- mean_shape_corr: `{baseline_payload['derived']['mean_shape_corr']:.6f}`",
        f"- mean_nrmse: `{baseline_payload['derived']['mean_nrmse']:.6f}`",
        f"- mean_phase_lag_cycles: `{baseline_payload['derived']['mean_phase_lag_cycles']:.6f}`",
        "",
    ]
    for attempt in attempts:
        lines.extend(
            [
                f"## Attempt: {attempt['label']}",
                "",
                f"- mean_shape_corr: `{attempt['updated_derived']['mean_shape_corr']:.6f}`",
                f"- mean_nrmse: `{attempt['updated_derived']['mean_nrmse']:.6f}`",
                f"- mean_phase_lag_cycles: `{attempt['updated_derived']['mean_phase_lag_cycles']:.6f}`",
                f"- mean_shape_corr_delta: `{attempt['delta']['mean_shape_corr_delta']:.6f}`",
                f"- mean_nrmse_delta: `{attempt['delta']['mean_nrmse_delta']:.6f}`",
                f"- mean_phase_lag_delta: `{attempt['delta']['mean_phase_lag_delta']:.6f}`",
                "",
                "### By Waveform",
                "",
            ]
        )
        for row in attempt["by_waveform"]:
            lines.extend(
                [
                    f"#### {row['waveform_type']}",
                    f"- case_count: `{row['case_count']}`",
                    f"- mean_shape_corr_delta: `{row['mean_shape_corr_delta']:.6f}`",
                    f"- mean_nrmse_delta: `{row['mean_nrmse_delta']:.6f}`",
                    f"- mean_phase_lag_delta: `{row['mean_phase_lag_delta']:.6f}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Conclusion",
            "",
            f"- best_attempt_label: `{payload['best_attempt_label']}`",
            f"- {conclusion}",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
