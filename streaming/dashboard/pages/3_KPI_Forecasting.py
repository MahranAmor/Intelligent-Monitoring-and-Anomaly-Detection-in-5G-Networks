import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styling import inject_style, PRIMARY_RED
from utils.api_client import get_forecast, get_cells
from utils.autorefresh import add_autorefresh_sidebar

st.set_page_config(page_title="KPI Forecasting", page_icon="📈", layout="wide")
inject_style()

add_autorefresh_sidebar("kpi_forecasting", default_seconds=60)

st.sidebar.markdown("### Filters")
cells = get_cells()
cell_options = ["All cells"] + (cells or [])
cell_choice = st.sidebar.selectbox("Cell", options=cell_options, index=0)
slice_choice = st.sidebar.selectbox(
    "Slice Type",
    options=["eMBB", "URLLC", "mMTC"],
    index=0,
    help="Threshold (red line) is the SLA contract for the chosen slice. "
         "Ignored when a specific cell is selected (each cell belongs to one slice).",
)

cell_id = None if cell_choice == "All cells" else cell_choice
# When a specific cell is selected, don't force a slice — each cell is fixed to one slice
slice_type = slice_choice if cell_id is None else None
model_param = "naive"

st.title("KPI Forecasting")

KPIS = [
    ("one_way_latency_ms",            "Pred Latency",    "ms",   "(ms)"),
    ("bler_percent",                  "Pred BLER",       "%",    "(%)"),
    ("throughput_dl_mbps",            "Pred Throughput", "Mbps", "DL (Mbps)"),
    ("handover_success_rate_percent", "Pred Handover",   "%",    "Success Rate (%)"),
]

forecasts = {
    kpi: get_forecast(kpi, cell_id=cell_id, slice_type=slice_type,
                      history_points=30, forecast_points=10, model=model_param)
    for kpi, _, _, _ in KPIS
}

cols = st.columns(4)
for col, (kpi, label, unit, _) in zip(cols, KPIS):
    f = forecasts[kpi]
    pred = f.get("predicted_value")
    status = f.get("sla_status", "unknown")
    badge_text, badge_color = {
        "within_threshold": ("Within threshold", "#2BB673"),
        "breach_predicted": ("Breach predicted", "#E50914"),
    }.get(status, ("--", "#888"))
    value_str = f"{pred:.1f} {unit}" if pred is not None else f"-- {unit}"
    with col:
        st.markdown(
            f"""
            <div style='background:#FFF;padding:1rem 1.25rem;border-radius:8px;
                        border:1px solid #EEE;box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
              <div style='color:#555;font-size:0.85rem;font-weight:600'>{label} Display</div>
              <div style='font-size:1.6rem;font-weight:700;margin:0.25rem 0'>{value_str}</div>
              <div style='color:{badge_color};font-weight:600'>{badge_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

row1_left, row1_right = st.columns(2)
row2_left, row2_right = st.columns(2)
charts = [row1_left, row1_right, row2_left, row2_right]


def draw_forecast_chart(container, title, f):
    historical = f.get("historical", [])
    forecast = f.get("forecast", [])
    threshold = f.get("threshold")

    with container:
        st.markdown(f"**{title}**")
        if not historical and not forecast:
            st.info("No data yet.")
            return

        fig = go.Figure()
        if historical:
            df_h = pd.DataFrame(historical)
            fig.add_trace(go.Scatter(
                x=list(range(len(df_h))),
                y=df_h["value"],
                mode="lines+markers",
                name="Historical",
                line=dict(color="#1f77b4", width=2),
            ))
        if forecast:
            df_f = pd.DataFrame(forecast)
            start = len(historical)
            fig.add_trace(go.Scatter(
                x=list(range(start, start + len(df_f))),
                y=df_f["value"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#9467bd", width=2, dash="dot"),
            ))
        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=PRIMARY_RED,
                annotation_text=f"SLA {threshold}",
                annotation_position="top left",
            )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None, yaxis_title=None,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)


for container, (kpi, _, _, title) in zip(charts, KPIS):
    nice_title = {
        "one_way_latency_ms": "Latency (ms)",
        "bler_percent": "BLER (%)",
        "throughput_dl_mbps": "Throughput DL (Mbps)",
        "handover_success_rate_percent": "Handover Success Rate (%)",
    }[kpi]
    draw_forecast_chart(container, nice_title, forecasts[kpi])

