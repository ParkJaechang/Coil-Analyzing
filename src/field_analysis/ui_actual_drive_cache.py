from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_actual_drive_cache_manager(
    cache_state: dict[str, dict[str, object]],
    cache_records: list[Any],
    build_selection_options: Any,
    fallback_selection: Any,
    edit_metadata: Any,
    delete_item: Any,
    *,
    selected_cache_key: str,
) -> None:
    st.markdown("#### 업로드된 validation 캐시")
    if not cache_records:
        st.caption("업로드된 validation 캐시 항목이 없습니다.")
        return
    rows = [
        {
            "내부 ID": record.cache_item_id,
            "cache_type": record.cache_type,
            "원본 파일명": record.original_filename,
            "표시 이름": record.display_name,
            "메모": record.user_note,
            "duplicate_of": record.duplicate_of,
            "source_path": record.source_path,
        }
        for record in cache_records
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    options, records_by_id, labels_by_id = build_selection_options(cache_records)
    selected_id = fallback_selection(options, st.session_state.get(selected_cache_key))
    if selected_id is None:
        st.session_state.pop(selected_cache_key, None)
        return
    st.session_state[selected_cache_key] = selected_id
    selected_id = st.selectbox(
        "validation 캐시 항목 선택",
        options=options,
        format_func=lambda cache_id: labels_by_id[cache_id],
        key=selected_cache_key,
    )
    selected = records_by_id[selected_id]
    st.caption(f"내부 ID: {selected.cache_item_id}")
    with st.expander("validation 캐시 metadata 편집/삭제", expanded=False):
        display_name = st.text_input(
            "표시 이름",
            value=selected.display_name,
            key=f"actual_drive_cache_display_name_{selected.cache_item_id}",
        )
        user_note = st.text_area(
            "메모",
            value=selected.user_note,
            key=f"actual_drive_cache_user_note_{selected.cache_item_id}",
        )
        if st.button("저장", key=f"save_actual_drive_cache_{selected.cache_item_id}"):
            edit_metadata(cache_state, selected.cache_item_id, display_name=display_name, user_note=user_note)
            st.rerun()
        st.caption("앱 캐시 목록에서 제거합니다. 임의의 원본 파일은 삭제하지 않습니다.")
        confirm_delete = st.checkbox("삭제 전 확인", key=f"confirm_actual_drive_cache_delete_{selected.cache_item_id}")
        if st.button(
            "선택한 validation 캐시 항목 삭제",
            key=f"delete_actual_drive_cache_{selected.cache_item_id}",
            disabled=not confirm_delete,
        ):
            delete_item(cache_state, selected.cache_item_id)
            st.session_state.pop(selected_cache_key, None)
            st.rerun()
