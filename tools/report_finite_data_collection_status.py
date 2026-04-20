from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


POLICY_DIR = Path("artifacts/policy_eval")
SCOPE_JSON = POLICY_DIR / "exact_and_finite_scope.json"
RECOMMEND_JSON = POLICY_DIR / "finite_data_recommendations.json"
OUTPUT_JSON = POLICY_DIR / "finite_data_collection_status.json"
OUTPUT_MD = POLICY_DIR / "finite_data_collection_status.md"


def main() -> int:
    scope = json.loads(SCOPE_JSON.read_text(encoding="utf-8"))
    recommend = json.loads(RECOMMEND_JSON.read_text(encoding="utf-8"))

    missing = scope.get("finite_exact_scope", {}).get("missing_combinations", [])
    recommendations = recommend.get("recommendations", [])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exact_recipe_count": int(scope.get("finite_exact_scope", {}).get("total_files", 0) or 0),
        "missing_exact_combinations": missing,
        "measurement_status": {
            "bench_confirmed_exact_recipe_count": 0,
            "new_exact_recipe_measured": [],
            "new_transient_support_measured": [],
        },
        "next_data_priority": recommendations,
        "conclusion": "finite는 generalization보다 exact recipe 갭 보완과 고주파 transient 데이터 보강이 더 빠른 경로입니다.",
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Finite Data Collection Status",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- exact_recipe_count: `{payload['exact_recipe_count']}`",
        f"- missing_exact_combinations: `{missing}`",
        "- bench_confirmed_exact_recipe_count: `0`",
        "- new_exact_recipe_measured: `[]`",
        "- new_transient_support_measured: `[]`",
        "",
        "## Next Data Priority",
        "",
    ]
    for idx, item in enumerate(recommendations, start=1):
        proposal = item.get("proposal", item)
        reason = item.get("reason") if isinstance(item, dict) else None
        if reason:
            lines.append(f"- {idx}. {proposal}")
            lines.append(f"  reason: `{reason}`")
        else:
            lines.append(f"- {idx}. {proposal}")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- {payload['conclusion']}",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
