import pandas as pd
import plotly.express as px
import streamlit as st

from utils.styling import inject_style, PRIMARY_RED, style_severity_df
from utils.api_client import get_anomaly_summary, get_anomaly_timeline_by_type, get_cells
from utils.autorefresh import add_autorefresh_sidebar

st.set_page_config(page_title="Anomaly Diagnosis", page_icon="🔍", layout="wide")
inject_style()

add_autorefresh_sidebar("anomaly_diagnosis", default_seconds=30)

st.sidebar.markdown("### Filters")
cells = ["Tout"] + get_cells()
cell_choice = st.sidebar.selectbox("Cell", options=cells, index=0)
slice_filter = st.sidebar.multiselect(
    "Slice Type",
    options=["eMBB", "mMTC", "URLLC"],
    default=["eMBB", "mMTC", "URLLC"],
)

cell_id = None if cell_choice == "Tout" else cell_choice
slice_type = slice_filter[0] if len(slice_filter) == 1 else None

st.title("Anomaly Diagnosis")

summary = get_anomaly_summary(slice_type=slice_type, cell_id=cell_id, hours=24)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Active Anomalies",    f"{summary.get('active_anomalies', 0):,}")
c2.metric("Dominant Anomaly Type", summary.get("dominant_anomaly_type") or "--")
c3.metric("Affected Cells",      f"{summary.get('affected_cells', 0):,}")
c4.metric("Affected UEs",        f"{summary.get('affected_ues', 0):,}")
c5.metric("Avg Anomaly Duration", f"{summary.get('avg_anomaly_duration_min', 0):,.0f} min")

st.markdown("")

left, right = st.columns([3, 2])

with left:
    st.markdown("**Anomaly Type Distribution**")
    types = summary.get("type_distribution", [])
    if not types:
        st.info("No anomaly types detected yet.")
    else:
        df = pd.DataFrame(types)
        df = df.sort_values("count", ascending=True)
        palette = px.colors.qualitative.Bold + px.colors.qualitative.Vivid
        fig = px.bar(
            df, x="count", y="ml_anomaly_type", orientation="h",
            color="ml_anomaly_type", color_discrete_sequence=palette,
        )
        fig.update_layout(
            height=380, showlegend=False,
            xaxis_title=None, yaxis_title=None,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("**Severity ranking**")
    ranking = summary.get("severity_ranking", [])
    if not ranking:
        st.info("Nothing to rank yet.")
    else:
        df = pd.DataFrame(ranking)[["cell_id", "dominant_anomaly_type", "severity"]]
        df.columns = ["cell_id", "Dominant Anomaly Type", "severity"]
        styled = style_severity_df(df, col="severity")
        st.dataframe(styled, hide_index=True, use_container_width=True, height=380)

st.markdown("**Anomaly type timeline**")
tl = get_anomaly_timeline_by_type(hours=24, slice_type=slice_type)
if not tl:
    st.info("No timeline data yet.")
else:
    df_tl = pd.DataFrame(tl)
    fig = px.bar(
        df_tl, x="hour", y="count", color="ml_anomaly_type",
        color_discrete_sequence=px.colors.qualitative.Bold + px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        barmode="stack", height=360,
        xaxis_title=None, yaxis_title=None,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
