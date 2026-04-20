from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


POLICY_DIR = Path("artifacts/policy_eval")
STAGE2_JSON = POLICY_DIR / "finite_generalization_stage2.json"
SCOPE_JSON = POLICY_DIR / "exact_and_finite_scope.json"
OUTPUT_JSON = POLICY_DIR / "finite_data_recommendations.json"
OUTPUT_MD = POLICY_DIR / "finite_data_recommendations.md"


def main() -> int:
    stage2 = json.loads(STAGE2_JSON.read_text(encoding="utf-8"))
    scope = json.loads(SCOPE_JSON.read_text(encoding="utf-8"))
    missing_exact = scope["finite_exact_scope"].get("missing_combinations", [])
    cases = pd.DataFrame(stage2.get("cases", []))
    by_freq: list[dict[str, object]] = []
    by_cycle: list[dict[str, object]] = []
    worst_cases: list[dict[str, object]] = []
    if not cases.empty:
        for freq_hz, group in cases.groupby("freq_hz", dropna=True, sort=True):
            by_freq.append(
                {
                    "freq_hz": float(freq_hz),
                    "shape_corr": float(pd.to_numeric(group["shape_corr"], errors="coerce").mean()),
                    "nrmse": float(pd.to_numeric(group["nrmse"], errors="coerce").mean()),
                    "phase_lag_s": float(pd.to_numeric(group["phase_lag_s"], errors="coerce").mean()),
                }
            )
        for cycle_count, group in cases.groupby("cycle_count", dropna=True, sort=True):
            by_cycle.append(
                {
                    "cycle_count": float(cycle_count),
                    "shape_corr": float(pd.to_numeric(group["shape_corr"], errors="coerce").mean()),
                    "nrmse": float(pd.to_numeric(group["nrmse"], errors="coerce").mean()),
                    "phase_lag_s": float(pd.to_numeric(group["phase_lag_s"], errors="coerce").mean()),
                }
            )
        worst_cases = (
            cases.sort_values("nrmse", ascending=False)
            .head(10)[["source_file", "freq_hz", "cycle_count", "target_current_pp", "shape_corr", "nrmse", "phase_lag_s"]]
            .to_dict(orient="records")
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_support_band_hz": scope.get("official_support_band_hz", {"min": 0.25, "max": 5.0}),
        "stage2_summary": stage2.get("summary", {}),
        "by_freq": by_freq,
        "by_cycle": by_cycle,
        "worst_cases": worst_cases,
        "missing_exact_combinations": missing_exact,
        "recommendations": [
            {
                "priority": 1,
                "type": "missing_exact_recipe",
                "proposal": "Measure the missing exact finite recipe: 1.0 Hz + 1.0 cycle + 20 pp.",
                "reason": "This is the only missing combination inside the current <= 5 Hz exact recipe table.",
            },
            {
                "priority": 2,
                "type": "high_freq_transient_support",
                "proposal": "Add exact finite transient data at 2.0 Hz and 5.0 Hz.",
                "reason": "Stage-2 preview quality remains weakest in the upper part of the official <= 5 Hz band.",
            },
            {
                "priority": 3,
                "type": "midband_repeatability",
                "proposal": "Repeat 0.75 cycle and 1.5 cycle measurements at 1.0 Hz and 1.25 Hz.",
                "reason": "Mid-band phase lag and preview consistency are still unstable across repeated finite runs.",
            },
        ],
        "conclusion": (
            "Finite should remain exact-recipe-first. The fastest improvement is to close the one missing exact recipe "
            "and strengthen 2/5 Hz transient coverage before revisiting any generalization work."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Finite Data Recommendations",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- official_support_band_hz: `{payload['official_support_band_hz']['min']} ~ {payload['official_support_band_hz']['max']}`",
        "",
        "## Stage-2 Preview Summary",
        "",
        f"- mean_shape_corr: `{payload['stage2_summary'].get('mean_shape_corr')}`",
        f"- mean_nrmse: `{payload['stage2_summary'].get('mean_nrmse')}`",
        f"- mean_phase_lag_s: `{payload['stage2_summary'].get('mean_phase_lag_s')}`",
        "",
        f"- missing_exact_combinations: `{missing_exact}`",
        "",
        "## Recommended Next Measurements",
        "",
    ]
    for item in payload["recommendations"]:
        lines.append(f"- P{item['priority']}: {item['proposal']} ({item['reason']})")
    lines.extend(["", f"- conclusion: `{payload['conclusion']}`"])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
