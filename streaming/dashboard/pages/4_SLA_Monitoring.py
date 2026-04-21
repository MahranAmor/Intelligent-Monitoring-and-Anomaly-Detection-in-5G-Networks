import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.styling import inject_style, PRIMARY_RED, SLICE_COLORS
from utils.api_client import get_sla_status, get_sla_breach_trend, get_alerts
from utils.autorefresh import add_autorefresh_sidebar

st.set_page_config(page_title="SLA Monitoring", page_icon="📊", layout="wide")
inject_style()

add_autorefresh_sidebar("sla_monitoring", default_seconds=30)

st.sidebar.markdown("### Filters")
slice_filter = st.sidebar.multiselect(
    "Slice Type",
    options=["eMBB", "mMTC", "URLLC"],
    default=["eMBB", "mMTC", "URLLC"],
)

st.title("SLA Monitoring")

sla_rows = get_sla_status()
by_slice = {r["slice_type"]: r for r in sla_rows if r.get("slice_type") in slice_filter}

SLICE_LABELS = {
    "eMBB":  ("enhanced mobile broadband",   "#1f77b4"),
    "URLLC": ("Ultra-reliable low latency",   "#E50914"),
    "mMTC":  ("Massive machine-type comms",   "#9467bd"),
}


def kpi_block(title, actual, threshold, unit):
    actual_str = f"{actual:.1f}" if actual is not None else "--"
    thresh_str = f"{threshold}" if threshold is not None else "--"
    return (
        f"<div style='color:#555;font-size:0.8rem'>{title}</div>"
        f"<div style='font-size:1.1rem;font-weight:700;margin-bottom:0.4rem'>"
        f"{actual_str} / {thresh_str} {unit}</div>"
    )


cols = st.columns(3)
for col, key in zip(cols, ["eMBB", "URLLC", "mMTC"]):
    label, color = SLICE_LABELS[key]
    row = by_slice.get(key)
    with col:
        html = (
            f"<div style='background:#FFF;padding:1rem 1.25rem;border-radius:8px;"
            f"border:1px solid #EEE;box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
            f"<div style='color:{color};font-weight:700;margin-bottom:0.5rem'>{label}</div>"
        )
        if row:
            kpis = row.get("kpi_status", {})
            tp = kpis.get("throughput_dl_mbps", {})
            lat = kpis.get("one_way_latency_ms", {})
            bler = kpis.get("bler_percent", {})
            ho = kpis.get("handover_success_rate_percent", {})
            html += kpi_block("Throughput DL", tp.get("value"), tp.get("threshold"), "Mbps")
            html += kpi_block("Latency",        lat.get("value"), lat.get("threshold"), "ms")
            html += kpi_block("BLER",           bler.get("value"), bler.get("threshold"), "%")
            html += kpi_block("Handover SR",    ho.get("value"), ho.get("threshold"), "%")
        else:
            html += "<div style='color:#888'>No data.</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

st.markdown("")

left, right = st.columns(2)

with left:
    st.markdown("**SLA compliance rate by slice (24h)**")
    if not by_slice:
        st.info("No SLA data yet.")
    else:
        categories = ["Throughput DL", "Handover SR", "Latency", "BLER"]
        fig = go.Figure()
        for key, row in by_slice.items():
            kpis = row.get("kpi_status", {})
            def compliance(v):
                if v is None or v.get("meets_sla") is None: return 0
                return 1.0 if v["meets_sla"] else 0.3
            vals = [
                compliance(kpis.get("throughput_dl_mbps")),
                compliance(kpis.get("handover_success_rate_percent")),
                compliance(kpis.get("one_way_latency_ms")),
                compliance(kpis.get("bler_percent")),
            ]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=f"{key} Compliance",
                line=dict(color=SLICE_COLORS.get(key, "#888")),
                opacity=0.5,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("**SLA breach trend (24h)**")
    trend = get_sla_breach_trend(hours=24)
    if not trend:
        st.info("No breaches recorded in the last 24h.")
    else:
        df_t = pd.DataFrame(trend)
        df_t = df_t[df_t["slice_type"].isin(slice_filter)]
        if df_t.empty:
            st.info("No breaches for the selected slice filter.")
        else:
            fig = px.line(
                df_t, x="bucket", y="breach_count", color="slice_type",
                markers=True, color_discrete_map=SLICE_COLORS,
            )
            fig.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title=None, yaxis_title=None,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("**Proactive Alerts**")
alerts = get_alerts(limit=50)
alerts = [a for a in alerts if a.get("slice_type") in slice_filter]
if not alerts:
    st.info("No active alerts.")
else:
    df_a = pd.DataFrame(alerts)
    df_a["alert_time"] = df_a["alert_time"].astype(str).str.slice(11, 16)
    display_cols = [c for c in ["alert_time", "slice_type", "kpi", "alert_message", "alert_action"]
                    if c in df_a.columns]
    df_a = df_a[display_cols].rename(columns={
        "alert_time": "Alert Time", "slice_type": "Slice", "kpi": "KPI",
        "alert_message": "Alert Message", "alert_action": "Alert Action",
    })
    st.dataframe(df_a, hide_index=True, use_container_width=True, height=260)
