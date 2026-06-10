from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import streamlit as st


StepStatus = Literal["ready", "next", "planned"]


@dataclass(frozen=True)
class WorkflowStep:
    key: str
    label: str
    status: StepStatus


V2_WORKFLOW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep("data", "Data", "ready"),
    WorkflowStep("first_model", "1st model", "next"),
    WorkflowStep("feedback", "Feedback", "planned"),
    WorkflowStep("export", "Export", "planned"),
)

FIRST_MODEL_POLICY_ROWS: tuple[dict[str, str], ...] = (
    {
        "cycle": "1.0",
        "peak_contract": "+peak1, -peak1",
        "modeling_basis": "phase sync -> peak-lobe gain envelope -> residual trim",
    },
    {
        "cycle": "1.5",
        "peak_contract": "+peak1, -peak1, +peak2",
        "modeling_basis": "phase sync -> peak-lobe gain envelope -> residual trim",
    },
)


def run_quick_lut_v2_app() -> None:
    st.set_page_config(
        page_title="Quick LUT v2",
        page_icon="Q2",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Quick LUT v2")
    st.caption("legacy app is preserved separately while the v2 workflow is rebuilt in smaller slices.")

    render_workflow_overview()
    data_tab, first_model_tab, feedback_tab, export_tab = st.tabs(
        ["Data", "1st model", "Feedback", "Export"]
    )
    with data_tab:
        render_data_stage()
    with first_model_tab:
        render_first_model_stage()
    with feedback_tab:
        render_placeholder_stage("Feedback")
    with export_tab:
        render_placeholder_stage("Export")


def render_workflow_overview() -> None:
    columns = st.columns(len(V2_WORKFLOW_STEPS))
    for column, step in zip(columns, V2_WORKFLOW_STEPS, strict=True):
        with column:
            st.metric(step.label, step.status)


def render_data_stage() -> None:
    st.subheader("Data")
    st.write("Upload memory and parsed measurement reuse will stay in shared core modules.")
    st.code("outputs/field_analysis_app_state/uploads", language="text")


def render_first_model_stage() -> None:
    st.subheader("1st model")
    st.dataframe(FIRST_MODEL_POLICY_ROWS, width="stretch", hide_index=True)
    st.write(
        "HallBz/raw field polarity is normalized before peak detection, and peak-lobe "
        "gains are calculated after the selected phase sync anchor."
    )
    st.write(
        "The production integration is reviewed in the legacy finite-first result panel for now; "
        "this v2 shell stays intentionally small until upload/model/export wiring is added."
    )


def render_placeholder_stage(name: str) -> None:
    st.subheader(name)
    st.write("This stage is intentionally empty in the first v2 shell PR.")
