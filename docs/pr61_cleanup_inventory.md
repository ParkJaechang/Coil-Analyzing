# PR61 Cleanup Inventory

This inventory records cleanup decisions for PR61. It is intentionally conservative: source/test files are not removed in this pass, generated data is not committed, and legacy UI is classified before any deletion.

## Cross-Repo Separation Policy

PR61 acceptance is limited to Streamlit WebApp / Quick LUT / `src.field_analysis` behavior.

- WinApp docs are historical cross-repo notes only. They must not be used as PR61 Streamlit acceptance evidence.
- AI/RL modeling app notes are experimental planning notes only. They must not be used as PR61 production acceptance evidence.
- AI/RL modeling and AI sweep dataset tooling have moved to the separate `ParkJaechang/Coil-Analyzing-AI` repository. PR61 does not contain AI/RL implementation and must not be used as AI repo acceptance evidence.

## Summary

| status | count | meaning |
|---|---:|---|
| keep | 13 | Required production, bridge, or source-of-truth file. |
| update | 4 | Stale status/policy document or PR-facing copy to refresh. |
| archive_candidate | 9 | Duplicate/stale report or legacy review artifact; do not delete without user approval. |
| external_reference_only | 4 | Cross-repo documentation that has no Streamlit runtime impact. |
| needs_review | 4 | Keep for now, but needs UX/module ownership decision. |

## Inventory

| path | type | current_status | reason | risk_if_removed | recommended_action | owner | runtime_impact | test_coverage |
|---|---|---|---|---|---|---|---|---|
| `docs/pr61_current_status.md` | doc | update | Stale head, CI state, and old +/-50mT / +/-5V policy. | PR readers follow wrong policy. | Updated to latest policy and source-of-truth pointer. | streamlit | none | doc only |
| `docs/pr61_acceptance_inventory.md` | doc | update | Acceptance snapshot may lag latest runtime policy. | User acceptance state becomes ambiguous. | Update checked head and CI/status notes when status pass completes. | streamlit | none | doc only |
| `docs/pr61_runtime_gap_report.md` | doc | archive_candidate | Historical runtime gap report; useful context but not current status. | Low if archived; moderate if deleted because it contains runtime evidence. | Keep, mark historical if edited later. | streamlit | none | doc only |
| `docs/pr61_next_agent_prompts.md` | doc | archive_candidate | Handoff prompt history, not source of truth. | Low; deleting loses context. | Keep as archive candidate. | streamlit | none | doc only |
| `implementation_acceptance_inventory.md` | doc | archive_candidate | Earlier root-level acceptance inventory; superseded by docs inventory and feedback log. | Low; deleting loses old audit trail. | Keep as archive candidate. | streamlit | none | doc only |
| `docs/pr61_user_feedback_resolution_log.md` | doc | keep | Current source of truth for user policies and repeated pitfalls. | High; future agents lose policy history. | Keep and update after each user feedback/fix cycle. | streamlit | none | doc only |
| `docs/winapp_core_dependency.md` | winapp_doc | external_reference_only | Historical WinApp dependency note; not PR61 Streamlit acceptance. | Confused as PR61 Streamlit acceptance source if unlabeled. | Keep only as historical cross-repo note; move to WinApp later if needed. | winapp | none in Streamlit | doc only |
| `docs/winapp_current_status.md` | winapp_doc | external_reference_only | Historical WinApp status; not PR61 Streamlit acceptance. | Confused as PR61 Streamlit acceptance source if unlabeled. | Keep only as historical cross-repo note; update in WinApp context only. | winapp | none in Streamlit | doc only |
| `docs/winapp_next_agent_prompts.md` | winapp_doc | external_reference_only | Historical WinApp handoff prompt; not PR61 Streamlit acceptance. | Confused as PR61 Streamlit acceptance source if unlabeled. | Keep only as historical cross-repo note or move to WinApp later. | winapp | none in Streamlit | doc only |
| `docs/winapp_runtime_checklist.md` | winapp_doc | external_reference_only | Historical WinApp runtime checklist; not PR61 Streamlit acceptance. | Confused as PR61 Streamlit acceptance source if unlabeled. | Keep only as historical cross-repo note or move to WinApp later. | winapp | none in Streamlit | doc only |
| `reports/COil_Analyzing_Canva_Result.md` | report | archive_candidate | Generated/report artifact, not runtime source. | Low. | Do not commit new generated variants; archive existing. | shared | none | none |
| `reports/COil_Analyzing_Development_Report.docx` | report | archive_candidate | Generated Word report currently dirty locally. | Low, but do not stage unrelated dirty report. | Leave uncommitted unless user explicitly requests report update. | shared | none | none |
| `reports/COil_Analyzing_Final_Deliverables_Summary.md` | report | archive_candidate | Generated summary currently dirty locally. | Low, but do not stage unrelated dirty report. | Leave uncommitted unless user explicitly requests report update. | shared | none | none |
| `docs/coil_analyzing_canva_report_final.md` | report | archive_candidate | Generated/final report doc, not runtime source. | Low. | Keep as archive candidate. | shared | none | none |
| `docs/coil_analyzing_docs_report_final.md` | report | archive_candidate | Generated/final docs report currently dirty locally. | Low, but do not stage unrelated dirty report. | shared | none | none |
| `docs/coil_analyzing_report_revision_summary.md` | report | archive_candidate | Report revision history. | Low. | Keep as archive candidate. | shared | none | none |
| `src/field_analysis/app_ui_snapshot.py` legacy hardware calibration block | legacy_ui | needs_review | Main UI must not expose DAQ/AMP/extrapolation as production modeling input. | Removing blindly can break debug/calibration diagnostics. | Keep Advanced/Legacy only; extract if it grows. | streamlit | possible UI only | UI contract tests |
| `src/field_analysis/ui_startup_compensation_review.py` | legacy_ui | needs_review | Startup review overlaps with residual/phase-sync review. | Removing blindly can break historical review path. | Decision: `keep_advanced_only` for now; merge/archive later. | streamlit | Advanced UI | limited |
| `src/field_analysis/ui_quick_lut_feedback.py` | source | keep | Legacy feedback selection still supports actual-drive discovery and review helpers. | High if removed. | Keep; continue hiding manual upload from main flow where possible. | streamlit | yes | covered |
| `src/field_analysis/ui_quick_lut_feedback_second_sources.py` | source | keep | Second-source discovery bridge. | High if removed. | Keep. | streamlit | yes | covered |
| `src/field_analysis/ui_modeling_error_summary.py` | source | keep | Error ratio summary UI for first/second modeling. | Medium/high. | Keep. | streamlit | yes | covered |
| `src/field_analysis/ui_quick_lut_target_debug.py` | source | keep | Target config debug UI. | Medium. | Keep collapsed/debug only. | streamlit | debug UI | covered |
| `launch_streamlit_with_free_port.cmd` | launcher | keep | Useful app launcher. | Low/medium. | Keep. | streamlit | local launcher | manual |
| `launch_streamlit_with_free_port_local.cmd` | launcher | keep | Local launcher variant. | Low/medium. | Keep. | streamlit | local launcher | manual |
| `src/field_analysis/finite_first_phase_sync.py` | source | keep | Production finite first phase-sync bridge. | High. | Do not delete; split only with tests. | streamlit | yes | covered |
| `src/field_analysis/first_modeling_voltage_response.py` | source | keep | First command voltage response helpers. | High. | Keep. | streamlit | yes | covered |
| `src/field_analysis/modeling_error_metrics.py` | source | keep | Error metrics shared with UI. | Medium/high. | Keep. | streamlit | yes | covered |
| `src/field_analysis/finite_first_normalization.py` | source | keep | Finite first normalization policy. | High. | Keep. | streamlit | yes | covered |
| `src/field_analysis/finite_phase_sync_math.py` | source | keep | Shared phase-sync math. | High. | Keep. | streamlit | yes | covered |
| `src/field_analysis/finite_phase_sync_support.py` | source | keep | Native support-window logic. | High. | Keep. | streamlit | yes | covered |
| `src/field_analysis/final_modeled_lut.py` | source | keep | Final LUT export contract. | High. | Keep. | streamlit | yes | covered |
| `src/field_analysis/voltage_policy.py` | source | keep | +/-10V command voltage source of truth. | High. | Keep and use everywhere. | streamlit | yes | covered |
| `src/field_analysis/target_templates.py` | source | keep | Analytic fixed rounded-triangle target source. | High. | Keep. | streamlit | yes | covered |
| `src/field_analysis/ai_sweep/*` | source | moved_external | AI sweep implementation moved to `ParkJaechang/Coil-Analyzing-AI`. | Low; PR61 must not host AI/RL implementation. | Removed from WebApp PR61. Continue work in AI repo. | ai_repo | none | AI repo tests |
| `tests/test_ai_sweep_*.py` | test | moved_external | AI sweep tests moved with implementation to `ParkJaechang/Coil-Analyzing-AI`. | Low. | Removed from WebApp PR61. Use AI repo CI as acceptance evidence for AI work. | ai_repo | none | AI repo tests |
| WinApp `src/coil_win_app/core_adapter.py` | source | keep | WinApp guarded Streamlit core bridge. | High in WinApp. | Keep in WinApp repo; do not modify from Streamlit cleanup. | winapp | yes in WinApp | WinApp tests |
| WinApp `src/coil_win_app/core_dependency.py` | source | keep | WinApp dependency SHA/version check. | High in WinApp. | Keep in WinApp repo; update only with WinApp PR. | winapp | yes in WinApp | WinApp tests |

## Cleanup Notes

- No user data, upload cache, export cache, CSV/XLSX output, build, dist, or local generated artifact should be staged by this cleanup pass.
- Do not delete production source/test files during PR61 cleanup. Prefer `archive_candidate`, `keep_advanced_only`, or explicit follow-up issue.
- The WinApp default `Data` / `Second_Result` folder policy should be tracked in the WinApp repo. Streamlit PR61 should only document the risk unless the WinApp files are present in the active workspace.
