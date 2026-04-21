import os
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def _get(path: str, params: dict | None = None, timeout: float = 10.0):
    url = f"{API_BASE_URL}{path}"
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


@st.cache_data(ttl=10)
def get_network_health():
    return _get("/api/network-health") or {"total_cells": 0, "healthy": 0, "degraded": 0, "critical": 0, "cells": []}


@st.cache_data(ttl=10)
def get_sla_status(slice_type: str | None = None):
    params = {"slice_type": slice_type} if slice_type else {}
    return _get("/api/sla-status", params) or []


@st.cache_data(ttl=10)
def get_alerts(slice_type=None, cell_id=None, severity=None, limit=200):
    params = {"limit": limit}
    if slice_type: params["slice_type"] = slice_type
    if cell_id:    params["cell_id"] = cell_id
    if severity:   params["severity"] = severity
    return _get("/api/alerts", params) or []


@st.cache_data(ttl=10)
def get_predictions(slice_type=None, cell_id=None, anomaly_only=False, anomaly_source="ml", limit=2000):
    params = {"limit": limit, "anomaly_only": anomaly_only, "anomaly_source": anomaly_source}
    if slice_type: params["slice_type"] = slice_type
    if cell_id:    params["cell_id"] = cell_id
    return _get("/api/predictions", params) or []


@st.cache_data(ttl=10)
def get_anomaly_summary(slice_type=None, cell_id=None, hours=24):
    params = {"hours": hours}
    if slice_type: params["slice_type"] = slice_type
    if cell_id:    params["cell_id"] = cell_id
    return _get("/api/anomaly-summary", params) or {
        "active_anomalies": 0, "dominant_anomaly_type": None, "affected_cells": 0,
        "avg_anomaly_duration_min": 0, "type_distribution": [], "severity_ranking": [],
    }


@st.cache_data(ttl=10)
def get_anomaly_timeline(hours=24, slice_type=None):
    params = {"hours": hours}
    if slice_type: params["slice_type"] = slice_type
    return _get("/api/anomaly-timeline", params) or []


@st.cache_data(ttl=10)
def get_anomaly_timeline_by_type(hours=24, slice_type=None):
    params = {"hours": hours}
    if slice_type: params["slice_type"] = slice_type
    return _get("/api/anomaly-timeline-by-type", params) or []


@st.cache_data(ttl=30)
def get_sla_breach_trend(hours=24):
    return _get("/api/sla-breach-trend", {"hours": hours}) or []


@st.cache_data(ttl=10)
def get_forecast(kpi: str, cell_id=None, slice_type=None, history_points=30, forecast_points=10, model="naive"):
    params = {"history_points": history_points, "forecast_points": forecast_points, "model": model}
    if cell_id:    params["cell_id"] = cell_id
    if slice_type: params["slice_type"] = slice_type
    return _get(f"/api/forecast/{kpi}", params) or {
        "historical": [], "forecast": [], "threshold": None,
        "predicted_value": None, "sla_status": "unknown",
    }


@st.cache_data(ttl=60)
def get_cells():
    return _get("/api/cells") or []
