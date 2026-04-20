from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


OFFICIAL_MAX_FREQ_HZ = 5.0
OUTPUT_MD = Path("artifacts/policy_eval/operational_support_scope.md")
OUTPUT_JSON = Path("artifacts/policy_eval/operational_support_scope.json")
SCOPE_JSON = Path("artifacts/policy_eval/exact_and_finite_scope.json")


def _load_scope() -> dict:
    if not SCOPE_JSON.exists():
        return {}
    return json.loads(SCOPE_JSON.read_text(encoding="utf-8"))


def _has_missing_exact(missing_exact: object) -> bool:
    if not isinstance(missing_exact, dict):
        return False
    for values in missing_exact.values():
        if isinstance(values, list) and values:
            return True
    return False


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    scope = _load_scope()
    finite_scope = scope.get("finite_all_exact_scope", {})
    sine_scope = scope.get("finite_exact_scope", {})
    triangle_scope = scope.get("finite_triangle_exact_scope", {})
    sine_total = int(sine_scope.get("total_files", 0))
    triangle_total = int(triangle_scope.get("total_files", 0))
    official_recipe_total = int(finite_scope.get("official_recipe_total", sine_total + triangle_total))
    missing_exact = finite_scope.get("missing_exact_combinations", {})
    provisional_preview = finite_scope.get("provisional_preview_combinations", [])

    finite_exact_lines = [f"finite exact recipe table: sine {sine_total} + triangle {triangle_total}, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz"]
    if _has_missing_exact(missing_exact):
        finite_exact_lines.append(f"remaining finite exact gaps: {missing_exact}")
    if provisional_preview:
        finite_exact_lines.append(
            "provisional preview fallback remains enabled for unsupported finite cells with an approved substitute recipe"
        )

    payload = {
        "generated_at_utc": generated_at,
        "official_support_band_hz": {"min": 0.25, "max": OFFICIAL_MAX_FREQ_HZ},
        "user_facing": {
            "supported_auto": [
                f"continuous / current / exact support only, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz",
                f"finite / exact-supported recipe only, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz",
            ],
            "supported_exact_pending_bench": [
                f"continuous / field / exact support only, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz (software-ready, bench pending)",
            ],
            "supported_finite_exact": [
                *finite_exact_lines,
            ],
            "preview_only": [
                "interpolated_in_hull requests",
                "finite requests outside the exact recipe table",
                f"continuous exact requests above {OFFICIAL_MAX_FREQ_HZ:g} Hz",
            ],
            "blocked": [
                "interpolated_edge",
                "out_of_hull",
                "field target auto",
            ],
            "experimental_reference": [
                f"continuous support above {OFFICIAL_MAX_FREQ_HZ:g} Hz remains reference-only",
            ],
            "fallback_rule": (
                "Unsupported requests stay in preview-only or blocked state and should show the nearest exact recipe "
                "within the <= 5 Hz official band when available."
            ),
            "recommended_next_measurements": [
                "continuous sine exact grid: 0.75 Hz @ 5/10/20 A",
                "continuous sine exact grid: 1.5 Hz @ 5/10/20 A",
                "continuous sine exact grid: 3.0 Hz @ 5/10/20 A",
                "continuous sine exact grid: 4.0 Hz @ 5/10/20 A",
                *(
                    ["finite missing exact: sine 1.0 Hz + 1.0 cycle + 20 pp"]
                    if _has_missing_exact(missing_exact)
                    else []
                ),
            ],
        },
        "internal_operations": {
            "current_exact_auto": {
                "status": "operational",
                "scope": f"continuous/current exact support only, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz",
                "policy_version": "v2",
            },
            "field_exact": {
                "status": "software_ready_bench_pending",
                "scope": f"continuous/field exact support only, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz",
                "remaining_gate": "bench smoke test sign-off",
            },
            "finite_exact": {
                "status": "software_ready_exact_recipe_only",
                "scope": f"finite exact-supported recipes only, <= {OFFICIAL_MAX_FREQ_HZ:g} Hz",
                "recipe_total": official_recipe_total,
                "recipe_breakdown": {"sine": sine_total, "triangle": triangle_total},
                "remaining_exact_gap": missing_exact,
                "provisional_preview": provisional_preview,
            },
            "interpolation": {
                "status": "closed",
                "scope": "preview-only",
                "reason": "interpolated auto remains disabled",
            },
            "finite_generalization": {
                "status": "preview_only",
                "scope": "research-only",
                "reason": "finite generalization stays preview-only",
            },
            "continuous_reference_above_band": {
                "status": "reference_only",
                "scope": f"> {OFFICIAL_MAX_FREQ_HZ:g} Hz continuous support",
                "reason": "excluded from official operation",
            },
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Operational Support Scope",
        "",
        f"- generated_at_utc: `{generated_at}`",
        f"- official_support_band_hz: `0.25 ~ {OFFICIAL_MAX_FREQ_HZ:g}`",
        "",
        "## User-Facing Scope",
        "",
        "### Auto Available",
        *[f"- {item}" for item in payload["user_facing"]["supported_auto"]],
        "",
        "### Exact, Bench Pending",
        *[f"- {item}" for item in payload["user_facing"]["supported_exact_pending_bench"]],
        "",
        "### Finite Exact Scope",
        *[f"- {item}" for item in payload["user_facing"]["supported_finite_exact"]],
        "",
        "### Preview Only",
        *[f"- {item}" for item in payload["user_facing"]["preview_only"]],
        "",
        "### Blocked",
        *[f"- {item}" for item in payload["user_facing"]["blocked"]],
        "",
        "### Reference Only",
        *[f"- {item}" for item in payload["user_facing"]["experimental_reference"]],
        "",
        f"- fallback_rule: `{payload['user_facing']['fallback_rule']}`",
        "",
        "### Recommended Next Measurements",
        *[f"- {item}" for item in payload["user_facing"]["recommended_next_measurements"]],
        "",
        "## Internal Operations",
        "",
        "| path | status | scope | note |",
        "| --- | --- | --- | --- |",
        f"| continuous/current exact | operational | <= {OFFICIAL_MAX_FREQ_HZ:g} Hz exact support only | policy v2 |",
        f"| continuous/field exact | software_ready_bench_pending | <= {OFFICIAL_MAX_FREQ_HZ:g} Hz exact support only | bench sign-off pending |",
        f"| finite exact | software_ready_exact_recipe_only | <= {OFFICIAL_MAX_FREQ_HZ:g} Hz exact-supported recipes only | sine {sine_total} + triangle {triangle_total} |",
        "| interpolation auto | closed | preview-only | disabled pending future R&D |",
        "| finite generalization | preview_only | research-only | preview quality not operational |",
        f"| continuous > {OFFICIAL_MAX_FREQ_HZ:g} Hz | reference_only | excluded from official operation | experimental/reference only |",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
