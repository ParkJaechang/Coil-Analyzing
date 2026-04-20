from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


POLICY_DIR = Path("artifacts/policy_eval")
OFFICIAL_MAX_FREQ_HZ = 5.0
ALL_BAND_CSV = POLICY_DIR / "policy_eval_v2_continuous_corpus_l1fo.csv"
ALL_BAND_JSON = POLICY_DIR / "policy_eval_v2_continuous_corpus_l1fo.json"
UPTO5_CSV = POLICY_DIR / "policy_eval_v2_upto5hz_continuous_corpus_l1fo.csv"
UPTO5_JSON = POLICY_DIR / "policy_eval_v2_upto5hz_continuous_corpus_l1fo.json"
SCOPE_JSON = POLICY_DIR / "exact_and_finite_scope.json"
OUTPUT_JSON = POLICY_DIR / "continuous_upto5hz_l1fo.json"
OUTPUT_MD = POLICY_DIR / "continuous_upto5hz_l1fo.md"


def _safe_float(value: object) -> float | None:
    series = pd.to_numeric(pd.Series([value]), errors="coerce")
    if series.isna().all():
        return None
    return float(series.iloc[0])


def _frame_metrics(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "case_count": 0,
            "auto_count": 0,
            "false_auto_count": 0,
            "false_block_count": 0,
            "mean_shape_corr": None,
            "mean_nrmse": None,
            "mean_phase_lag_cycles": None,
            "mean_predicted_error_band": None,
            "support_state_counts": {},
        }
    return {
        "case_count": int(len(frame)),
        "auto_count": int(pd.to_numeric(frame["allow_auto_recommendation"], errors="coerce").fillna(False).astype(bool).sum())
        if "allow_auto_recommendation" in frame.columns
        else 0,
        "false_auto_count": int((frame.get("confusion_label") == "false_auto").sum()) if "confusion_label" in frame.columns else 0,
        "false_block_count": int((frame.get("confusion_label") == "false_block").sum()) if "confusion_label" in frame.columns else 0,
        "mean_shape_corr": _safe_float(pd.to_numeric(frame.get("realized_shape_corr"), errors="coerce").mean()),
        "mean_nrmse": _safe_float(pd.to_numeric(frame.get("realized_nrmse"), errors="coerce").mean()),
        "mean_phase_lag_cycles": _safe_float(pd.to_numeric(frame.get("realized_phase_lag_cycles"), errors="coerce").mean()),
        "mean_predicted_error_band": _safe_float(pd.to_numeric(frame.get("predicted_error_band"), errors="coerce").mean()),
        "support_state_counts": frame.get("support_state", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
    }


def main() -> int:
    all_band_summary = json.loads(ALL_BAND_JSON.read_text(encoding="utf-8"))
    upto5_summary = json.loads(UPTO5_JSON.read_text(encoding="utf-8"))
    scope_payload = json.loads(SCOPE_JSON.read_text(encoding="utf-8"))

    all_band_frame = pd.read_csv(ALL_BAND_CSV, encoding="utf-8-sig")
    all_band_subset = all_band_frame[
        pd.to_numeric(all_band_frame["target_freq_hz"], errors="coerce") <= OFFICIAL_MAX_FREQ_HZ
    ].copy()
    upto5_frame = pd.read_csv(UPTO5_CSV, encoding="utf-8-sig")

    pre_metrics = _frame_metrics(all_band_subset)
    post_metrics = _frame_metrics(upto5_frame)
    delta = {
        "case_count_delta": int(post_metrics["case_count"]) - int(pre_metrics["case_count"]),
        "mean_shape_corr_delta": (
            float(post_metrics["mean_shape_corr"]) - float(pre_metrics["mean_shape_corr"])
            if pre_metrics["mean_shape_corr"] is not None and post_metrics["mean_shape_corr"] is not None
            else None
        ),
        "mean_nrmse_delta": (
            float(post_metrics["mean_nrmse"]) - float(pre_metrics["mean_nrmse"])
            if pre_metrics["mean_nrmse"] is not None and post_metrics["mean_nrmse"] is not None
            else None
        ),
        "mean_phase_lag_cycles_delta": (
            float(post_metrics["mean_phase_lag_cycles"]) - float(pre_metrics["mean_phase_lag_cycles"])
            if pre_metrics["mean_phase_lag_cycles"] is not None and post_metrics["mean_phase_lag_cycles"] is not None
            else None
        ),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_support_band_hz": {"min": 0.25, "max": OFFICIAL_MAX_FREQ_HZ},
        "official_continuous_scope": scope_payload.get("continuous_official_exact_scope", {}),
        "all_band_subset_reference": {
            "source_policy_version": all_band_summary.get("policy_version"),
            "metrics": pre_metrics,
        },
        "upto5hz_refit": {
            "source_policy_version": upto5_summary.get("policy_version"),
            "metrics": post_metrics,
        },
        "delta": delta,
        "bench_case_proposal": [
            {
                "case": "continuous exact current",
                "recommended_freq_hz": 0.5,
                "note": "inside official band and already matched by the validated exact-current export bundle",
            },
            {
                "case": "continuous exact field",
                "recommended_freq_hz": 0.25,
                "note": "inside official band and already matched by the validated exact-field export bundle",
            },
            {
                "case": "finite exact sine",
                "recommended_freq_hz": 0.5,
                "recommended_cycles": 1.0,
                "recommended_level_pp_a": 20,
                "note": "inside official band and already matched by the validated finite exact export bundle",
            },
        ],
        "conclusion": (
            "Narrowing the steady-state evaluation corpus to <= 5 Hz improves product focus, "
            "but interpolated preview quality still needs model-side improvement before any auto rollout can be reconsidered."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Continuous <= 5 Hz L1FO Re-Evaluation",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- official_support_band_hz: `0.25 ~ {OFFICIAL_MAX_FREQ_HZ:g}`",
        "",
        "## Official Continuous Scope",
        "",
        f"- usable_auto: `{scope_payload['continuous_official_exact_scope']['operational_status']['usable_auto']}`",
        f"- preview_only: `{scope_payload['continuous_official_exact_scope']['operational_status']['preview_only']}`",
        "",
        "## Model-Side Compare (Pre vs Post <= 5 Hz Refit)",
        "",
        "| dataset | case_count | auto_count | false_auto | false_block | mean_shape_corr | mean_nrmse | mean_phase_lag_cycles |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        "| all-band subset reference | "
        + " | ".join(
            str(pre_metrics[key])
            for key in [
                "case_count",
                "auto_count",
                "false_auto_count",
                "false_block_count",
                "mean_shape_corr",
                "mean_nrmse",
                "mean_phase_lag_cycles",
            ]
        )
        + " |",
        "| <=5 Hz refit | "
        + " | ".join(
            str(post_metrics[key])
            for key in [
                "case_count",
                "auto_count",
                "false_auto_count",
                "false_block_count",
                "mean_shape_corr",
                "mean_nrmse",
                "mean_phase_lag_cycles",
            ]
        )
        + " |",
        "",
        f"- delta(mean_shape_corr): `{delta['mean_shape_corr_delta']}`",
        f"- delta(mean_nrmse): `{delta['mean_nrmse_delta']}`",
        f"- delta(mean_phase_lag_cycles): `{delta['mean_phase_lag_cycles_delta']}`",
        "",
        "## Bench Case Proposal (<= 5 Hz)",
        "",
    ]
    for case in payload["bench_case_proposal"]:
        parts = [f"freq={case['recommended_freq_hz']:g} Hz"]
        if "recommended_cycles" in case:
            parts.append(f"cycles={case['recommended_cycles']:g}")
        if "recommended_level_pp_a" in case:
            parts.append(f"level={case['recommended_level_pp_a']} pp")
        lines.append(f"- {case['case']}: " + ", ".join(parts) + f" ({case['note']})")
    lines.extend(["", f"- conclusion: `{payload['conclusion']}`"])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
