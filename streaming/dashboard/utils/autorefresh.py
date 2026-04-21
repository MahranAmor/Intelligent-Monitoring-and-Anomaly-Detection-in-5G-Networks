import streamlit as st
from streamlit_autorefresh import st_autorefresh


def add_autorefresh_sidebar(page_key: str, default_seconds: int = 30):
    st.sidebar.markdown("### Auto-refresh")
    enabled = st.sidebar.toggle("Enabled", value=True, key=f"auto_{page_key}")
    interval = st.sidebar.select_slider(
        "Interval",
        options=[15, 30, 60, 120, 300],
        value=default_seconds,
        format_func=lambda s: f"{s}s",
        key=f"interval_{page_key}",
    )
    if enabled:
        st_autorefresh(interval=interval * 1000, key=f"tick_{page_key}")
    if st.sidebar.button("Refresh now", key=f"refresh_now_{page_key}"):
        st.cache_data.clear()
        st.rerun()
