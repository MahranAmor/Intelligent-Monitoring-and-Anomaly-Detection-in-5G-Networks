import os
import sys
import json
import time
from collections import deque

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC_PREDICTIONS = os.environ.get("KAFKA_TOPIC_PREDICTIONS", "predictions")

st.set_page_config(
    page_title="5G Network Anomaly Monitor",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def get_kafka_consumer():
    for attempt in range(1, 21):
        try:
            consumer = KafkaConsumer(
                TOPIC_PREDICTIONS,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                group_id="streamlit-dashboard",
                auto_offset_reset="latest",
                consumer_timeout_ms=1000,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            return consumer
        except NoBrokersAvailable:
            time.sleep(3)
    return None


# Session state initialization
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=500)
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0
if "total_anomalies" not in st.session_state:
    st.session_state.total_anomalies = 0
if "correct_predictions" not in st.session_state:
    st.session_state.correct_predictions = 0

# Sidebar
st.sidebar.title("Controls")
auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh interval (s)", min_value=1, max_value=10, value=3)
slice_filter = st.sidebar.multiselect(
    "Slice type filter",
    options=["eMBB", "URLLC", "mMTC"],
    default=["eMBB", "URLLC", "mMTC"],
)
latency_threshold = st.sidebar.number_input("Latency threshold (ms)", value=50.0)
packet_loss_threshold = st.sidebar.number_input("Packet loss threshold (%)", value=2.0)

# Consume new messages
consumer = get_kafka_consumer()
if consumer is not None:
    try:
        for msg in consumer:
            data = msg.value
            st.session_state.buffer.append(data)
            st.session_state.total_processed += 1
            if data.get("ml_prediction") == 1:
                st.session_state.total_anomalies += 1
            if data.get("ml_prediction") == data.get("actual_anomaly"):
                st.session_state.correct_predictions += 1
    except Exception:
        pass

# Main content
st.title("📡 5G Network Anomaly Monitor")

buffer_list = list(st.session_state.buffer)

if not buffer_list:
    st.info("Waiting for data from Kafka...")
else:
    df = pd.DataFrame(buffer_list)

    # Apply slice filter
    if slice_filter:
        df = df[df["slice_type"].isin(slice_filter)]

    total = st.session_state.total_processed
    anomalies = st.session_state.total_anomalies
    correct = st.session_state.correct_predictions
    anomaly_rate = (anomalies / total * 100) if total > 0 else 0.0
    accuracy = (correct / total * 100) if total > 0 else 0.0
    avg_latency = df["one_way_latency_ms"].mean() if "one_way_latency_ms" in df.columns else 0.0
    avg_throughput = df["throughput_dl_mbps"].mean() if "throughput_dl_mbps" in df.columns else 0.0

    # Row 1 — KPI Summary
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Processed", f"{total:,}")
    col2.metric("Anomalies Detected", f"{anomalies:,}", delta=f"{anomaly_rate:.1f}%")
    col3.metric("Model Accuracy", f"{accuracy:.1f}%")
    col4.metric("Avg Latency", f"{avg_latency:.2f} ms")
    col5.metric("Avg Throughput DL", f"{avg_throughput:.2f} Mbps")

    st.divider()

    # Row 2 — KPI Time Series
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("One-Way Latency over Time")
        fig_lat = go.Figure()
        for stype in df["slice_type"].unique():
            sdf = df[df["slice_type"] == stype]
            fig_lat.add_trace(go.Scatter(
                x=list(range(len(sdf))),
                y=sdf["one_way_latency_ms"],
                mode="lines",
                name=stype,
            ))
        fig_lat.add_hline(
            y=latency_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold {latency_threshold} ms",
        )
        fig_lat.update_layout(height=350, xaxis_title="Message index", yaxis_title="Latency (ms)")
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_right:
        st.subheader("Downlink Throughput over Time")
        fig_tp = go.Figure()
        for stype in df["slice_type"].unique():
            sdf = df[df["slice_type"] == stype]
            fig_tp.add_trace(go.Scatter(
                x=list(range(len(sdf))),
                y=sdf["throughput_dl_mbps"],
                mode="lines",
                name=stype,
            ))
        fig_tp.update_layout(height=350, xaxis_title="Message index", yaxis_title="Throughput (Mbps)")
        st.plotly_chart(fig_tp, use_container_width=True)

    st.divider()

    # Row 3 — Anomaly Analysis
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Normal vs Anomaly")
        pred_counts = df["ml_anomaly_label"].value_counts().reset_index()
        pred_counts.columns = ["label", "count"]
        fig_pie = px.pie(pred_counts, names="label", values="count", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Anomaly Types Detected")
        if "actual_anomaly_type" in df.columns:
            type_counts = df[df["ml_prediction"] == 1]["actual_anomaly_type"].value_counts().head(8).reset_index()
            type_counts.columns = ["anomaly_type", "count"]
            fig_bar = px.bar(
                type_counts, x="count", y="anomaly_type", orientation="h",
                color="count", color_continuous_scale="Reds",
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_c:
        st.subheader("Confidence Distribution")
        if "ml_confidence" in df.columns:
            fig_hist = px.histogram(
                df, x="ml_confidence", color="ml_anomaly_label", nbins=30,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Row 4 — Per-Cell Anomaly Heatmap
    st.subheader("Top 20 Cells by Anomaly Rate")
    if "cell_id" in df.columns:
        cell_stats = df.groupby("cell_id").agg(
            anomaly_count=("ml_prediction", "sum"),
            total=("ml_prediction", "count"),
        ).reset_index()
        cell_stats["anomaly_rate"] = cell_stats["anomaly_count"] / cell_stats["total"]
        cell_stats["avg_latency"] = df.groupby("cell_id")["one_way_latency_ms"].mean().values
        top20 = cell_stats.nlargest(20, "anomaly_rate")
        fig_cell = px.bar(
            top20, x="cell_id", y="anomaly_rate",
            color="anomaly_rate", color_continuous_scale="RdYlGn_r",
        )
        fig_cell.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cell, use_container_width=True)

    st.divider()

    # Row 5 — Recent Anomaly Alerts
    st.subheader("Recent Anomaly Alerts (last 10)")
    anomaly_df = df[df["ml_prediction"] == 1].tail(10)
    display_cols = [
        c for c in [
            "timestamp", "cell_id", "slice_type", "one_way_latency_ms",
            "throughput_dl_mbps", "ml_confidence", "actual_anomaly_type",
        ] if c in anomaly_df.columns
    ]
    if not anomaly_df.empty:
        st.dataframe(anomaly_df[display_cols], hide_index=True)
    else:
        st.info("No anomalies detected yet.")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
