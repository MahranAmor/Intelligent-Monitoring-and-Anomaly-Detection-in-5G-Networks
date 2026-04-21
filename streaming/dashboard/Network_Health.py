import pandas as pd
import plotly.express as px
import streamlit as st

from utils.styling import inject_style, PRIMARY_RED, SLICE_COLORS
from utils.api_client import get_network_health, get_anomaly_timeline, get_predictions
from utils.autorefresh import add_autorefresh_sidebar

st.set_page_config(
    page_title="5G Network Anomaly Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_style()

add_autorefresh_sidebar("network_health", default_seconds=30)

st.sidebar.markdown("### Filters")
slice_filter = st.sidebar.multiselect(
    "Slice Type",
    options=["eMBB", "mMTC", "URLLC"],
    default=["eMBB", "mMTC", "URLLC"],
)

st.title("Network Health")

health = get_network_health()
total = health.get("total_cells", 0)
healthy = health.get("healthy", 0)
degraded = health.get("degraded", 0)
critical = health.get("critical", 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cells", f"{total}")
c2.metric("Healthy", f"{healthy}")
c3.metric("Degraded", f"{degraded}")
c4.metric("Critical", f"{critical}")

st.markdown("")

left, right = st.columns([3, 2])

with left:
    st.markdown("**Top 10 critical cells**")
    cells = health.get("cells", [])
    if not cells:
        st.info("No cell data yet.")
    else:
        df_cells = pd.DataFrame(cells)
        df_cells = df_cells.sort_values("anomaly_count", ascending=False).head(10)
        fig = px.bar(
            df_cells,
            x="anomaly_count",
            y="cell_id",
            orientation="h",
            color="anomaly_rate",
            color_continuous_scale=[[0, "#F5A623"], [1, PRIMARY_RED]],
            labels={"anomaly_count": "Critical Anomaly Count", "cell_id": "cell_id"},
        )
        fig.update_layout(
            height=360,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("**Anomalies by slice**")
    rows = get_predictions(anomaly_only=True, anomaly_source="actual", limit=5000)
    if not rows:
        st.info("No anomalies detected yet.")
    else:
        df = pd.DataFrame(rows)
        if slice_filter:
            df = df[df["slice_type"].isin(slice_filter)]
        if df.empty:
            st.info("No anomalies for the selected slice filter.")
        else:
            counts = df["slice_type"].value_counts().reset_index()
            counts.columns = ["slice_type", "count"]
            fig = px.pie(
                counts, names="slice_type", values="count", hole=0.55,
                color="slice_type", color_discrete_map=SLICE_COLORS,
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, use_container_width=True)

st.markdown("**Anomaly timeline (last 24h)**")
tl = get_anomaly_timeline(hours=24)
if not tl:
    st.info("No timeline data yet.")
else:
    df_tl = pd.DataFrame(tl)
    if slice_filter:
        df_tl = df_tl[df_tl["slice_type"].isin(slice_filter)]
    if df_tl.empty:
        st.info("No data for the selected slice filter.")
    else:
        agg = df_tl.groupby("hour", as_index=False)["anomaly_count"].sum()
        fig = px.area(
            agg, x="hour", y="anomaly_count",
            color_discrete_sequence=[PRIMARY_RED],
        )
        fig.update_traces(fillcolor="rgba(229,9,20,0.25)")
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None, yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
