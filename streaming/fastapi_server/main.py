import os
import json
import time
import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC_PREDICTIONS = os.environ.get("KAFKA_TOPIC_PREDICTIONS", "predictions")
DB_PATH = os.environ.get("DB_PATH", "/app/data/5g_monitor.db")

# SLA thresholds per slice type
SLA_THRESHOLDS = {
    "eMBB":  {"throughput_dl_mbps": 100.0, "one_way_latency_ms": 20.0,  "bler_percent": 5.0,  "handover_success_rate_percent": 95.0},
    "URLLC": {"throughput_dl_mbps": 0.5,   "one_way_latency_ms": 100.0, "bler_percent": 10.0, "handover_success_rate_percent": 90.0},
    "mMTC":  {"throughput_dl_mbps": 50.0,  "one_way_latency_ms": 1.0,   "bler_percent": 0.1,  "handover_success_rate_percent": 99.9},
}

ALERT_ACTIONS = {
    "throughput_dl_mbps": "Check backhaul capacity",
    "one_way_latency_ms": "Check routing / congestion",
    "bler_percent": "Check radio interference",
    "handover_success_rate_percent": "Check handover config",
}

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cell_id TEXT,
            ue_id TEXT,
            slice_type TEXT,
            latitude REAL,
            longitude REAL,
            one_way_latency_ms REAL,
            jitter_ms REAL,
            rtt_ms REAL,
            throughput_dl_mbps REAL,
            throughput_ul_mbps REAL,
            packet_loss_percent REAL,
            reliability_percent REAL,
            bler_percent REAL,
            handover_success_rate_percent REAL,
            energy_efficiency_bits_per_joule REAL,
            ml_prediction INTEGER,
            ml_confidence REAL,
            ml_anomaly_label TEXT,
            ml_anomaly_type TEXT,
            ml_anomaly_type_confidence REAL,
            actual_anomaly INTEGER,
            actual_anomaly_type TEXT,
            ingested_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_time TEXT,
            cell_id TEXT,
            slice_type TEXT,
            kpi TEXT,
            kpi_value REAL,
            threshold REAL,
            alert_message TEXT,
            alert_action TEXT,
            severity TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # Back-compat: add ue_id column to existing DBs.
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "ue_id" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN ue_id TEXT")
    conn.commit()
    conn.close()
    print("[FastAPI] Database initialised.", flush=True)


def insert_prediction(conn: sqlite3.Connection, record: dict):
    conn.execute("""
        INSERT INTO predictions (
            timestamp, cell_id, ue_id, slice_type, latitude, longitude,
            one_way_latency_ms, jitter_ms, rtt_ms,
            throughput_dl_mbps, throughput_ul_mbps,
            packet_loss_percent, reliability_percent, bler_percent,
            handover_success_rate_percent, energy_efficiency_bits_per_joule,
            ml_prediction, ml_confidence, ml_anomaly_label,
            ml_anomaly_type, ml_anomaly_type_confidence,
            actual_anomaly, actual_anomaly_type
        ) VALUES (
            :timestamp, :cell_id, :ue_id, :slice_type, :latitude, :longitude,
            :one_way_latency_ms, :jitter_ms, :rtt_ms,
            :throughput_dl_mbps, :throughput_ul_mbps,
            :packet_loss_percent, :reliability_percent, :bler_percent,
            :handover_success_rate_percent, :energy_efficiency_bits_per_joule,
            :ml_prediction, :ml_confidence, :ml_anomaly_label,
            :ml_anomaly_type, :ml_anomaly_type_confidence,
            :actual_anomaly, :actual_anomaly_type
        )
    """, {k: record.get(k) for k in [
        "timestamp", "cell_id", "ue_id", "slice_type", "latitude", "longitude",
        "one_way_latency_ms", "jitter_ms", "rtt_ms",
        "throughput_dl_mbps", "throughput_ul_mbps",
        "packet_loss_percent", "reliability_percent", "bler_percent",
        "handover_success_rate_percent", "energy_efficiency_bits_per_joule",
        "ml_prediction", "ml_confidence", "ml_anomaly_label",
        "ml_anomaly_type", "ml_anomaly_type_confidence",
        "actual_anomaly", "actual_anomaly_type",
    ]})
    conn.commit()


def evaluate_sla_and_alert(conn: sqlite3.Connection, record: dict):
    slice_type = record.get("slice_type", "")
    thresholds = SLA_THRESHOLDS.get(slice_type)
    if not thresholds:
        return

    cell_id = record.get("cell_id", "")
    ts = record.get("timestamp", datetime.utcnow().isoformat())

    kpi_checks = [
        ("throughput_dl_mbps",           lambda v, t: v < t,  "low"),
        ("one_way_latency_ms",            lambda v, t: v > t,  "high"),
        ("bler_percent",                  lambda v, t: v > t,  "high"),
        ("handover_success_rate_percent", lambda v, t: v < t,  "low"),
    ]

    for kpi, breach_fn, direction in kpi_checks:
        value = record.get(kpi)
        threshold = thresholds.get(kpi)
        if value is None or threshold is None:
            continue
        if breach_fn(float(value), threshold):
            proximity = abs(float(value) - threshold) / threshold if threshold != 0 else 0
            severity = "Critical" if proximity > 0.2 else "Low"
            direction_word = "below" if direction == "low" else "above"
            msg = f"{slice_type} {kpi} {float(value):.1f} {direction_word} SLA {threshold}"
            action = ALERT_ACTIONS.get(kpi, "Investigate")
            conn.execute("""
                INSERT INTO alerts (alert_time, cell_id, slice_type, kpi, kpi_value, threshold,
                                    alert_message, alert_action, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ts, cell_id, slice_type, kpi, float(value), threshold, msg, action, severity))
    conn.commit()


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []
        self._lock = threading.Lock()

    def add(self, ws: WebSocket):
        with self._lock:
            self.active.append(ws)

    def remove(self, ws: WebSocket):
        with self._lock:
            self.active = [c for c in self.active if c is not ws]

    async def broadcast(self, data: dict):
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove(ws)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Kafka consumer background thread
# ---------------------------------------------------------------------------

def kafka_consumer_thread():
    print("[FastAPI] Kafka consumer thread starting...", flush=True)
    consumer = None
    for attempt in range(1, 31):
        try:
            consumer = KafkaConsumer(
                TOPIC_PREDICTIONS,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                group_id="fastapi-monitor",
                auto_offset_reset="earliest",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            print(f"[FastAPI] Connected to Kafka on attempt {attempt}", flush=True)
            break
        except NoBrokersAvailable:
            print(f"[FastAPI] Kafka not ready (attempt {attempt}/30), retrying...", flush=True)
            time.sleep(5)

    if consumer is None:
        print("[FastAPI] Could not connect to Kafka. Consumer thread exiting.", flush=True)
        return

    db_conn = get_db()
    count = 0

    for msg in consumer:
        record = msg.value
        try:
            insert_prediction(db_conn, record)
            evaluate_sla_and_alert(db_conn, record)

            import asyncio
            loop = app.state.loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(manager.broadcast(record), loop)

            count += 1
            if count % 500 == 0:
                print(f"[FastAPI] Stored {count} records.", flush=True)
        except Exception as e:
            print(f"[FastAPI] Error processing record: {e}", flush=True)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    app.state.loop = asyncio.get_event_loop()
    init_db()
    t = threading.Thread(target=kafka_consumer_thread, daemon=True)
    t.start()
    print("[FastAPI] Server ready.", flush=True)
    yield


app = FastAPI(title="5G Anomaly Monitor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rows_to_dicts(rows) -> List[dict]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/predictions")
def get_predictions(
    slice_type: Optional[str] = None,
    cell_id: Optional[str] = None,
    anomaly_only: bool = False,
    anomaly_source: str = "ml",  # "ml" or "actual"
    limit: int = Query(default=1000, le=50000),
    offset: int = 0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    conn = get_db()
    clauses, params = [], []

    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    if cell_id:
        clauses.append("cell_id = ?")
        params.append(cell_id)
    if anomaly_only:
        col = "actual_anomaly" if anomaly_source == "actual" else "ml_prediction"
        clauses.append(f"{col} = 1")
    if start_time:
        clauses.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        clauses.append("timestamp <= ?")
        params.append(end_time)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params += [limit, offset]
    rows = conn.execute(
        f"SELECT * FROM predictions {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        params,
    ).fetchall()
    conn.close()
    return rows_to_dicts(rows)


@app.get("/api/network")
def get_network(
    slice_type: Optional[str] = None,
    cell_id: Optional[str] = None,
    limit: int = Query(default=1000, le=50000),
    offset: int = 0,
):
    conn = get_db()
    clauses, params = [], []
    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    if cell_id:
        clauses.append("cell_id = ?")
        params.append(cell_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params += [limit, offset]
    rows = conn.execute(
        f"""SELECT timestamp, cell_id, slice_type, latitude, longitude,
                   one_way_latency_ms, jitter_ms, rtt_ms,
                   throughput_dl_mbps, throughput_ul_mbps,
                   packet_loss_percent, bler_percent, handover_success_rate_percent
            FROM predictions {where}
            ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
        params,
    ).fetchall()
    conn.close()
    return rows_to_dicts(rows)


@app.get("/api/alerts")
def get_alerts(
    slice_type: Optional[str] = None,
    cell_id: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(default=500, le=10000),
    offset: int = 0,
):
    conn = get_db()
    clauses, params = [], []
    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    if cell_id:
        clauses.append("cell_id = ?")
        params.append(cell_id)
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params += [limit, offset]
    rows = conn.execute(
        f"SELECT * FROM alerts {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params,
    ).fetchall()
    conn.close()
    return rows_to_dicts(rows)


@app.get("/api/sla-status")
def get_sla_status(
    slice_type: Optional[str] = None,
):
    conn = get_db()
    clauses, params = [], []
    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = conn.execute(
        f"""SELECT
                slice_type,
                COUNT(*) AS total_records,
                AVG(throughput_dl_mbps) AS avg_throughput_dl,
                AVG(one_way_latency_ms) AS avg_latency,
                AVG(bler_percent) AS avg_bler,
                AVG(handover_success_rate_percent) AS avg_handover_sr,
                SUM(ml_prediction) AS total_anomalies
            FROM predictions {where}
            GROUP BY slice_type""",
        params,
    ).fetchall()
    conn.close()

    result = []
    for row in rows:
        d = dict(row)
        st = d["slice_type"]
        thresholds = SLA_THRESHOLDS.get(st, {})

        kpi_map = {
            "throughput_dl_mbps": ("avg_throughput_dl", lambda v, t: v >= t),
            "one_way_latency_ms": ("avg_latency",       lambda v, t: v <= t),
            "bler_percent":       ("avg_bler",           lambda v, t: v <= t),
            "handover_success_rate_percent": ("avg_handover_sr", lambda v, t: v >= t),
        }
        kpi_status = {}
        for kpi, (col, meets_fn) in kpi_map.items():
            val = d.get(col)
            thresh = thresholds.get(kpi)
            kpi_status[kpi] = {
                "value": round(val, 4) if val is not None else None,
                "threshold": thresh,
                "meets_sla": bool(meets_fn(val, thresh)) if val is not None and thresh is not None else None,
            }

        d["kpi_status"] = kpi_status
        d["thresholds"] = thresholds
        result.append(d)

    return result


@app.get("/api/network-health")
def get_network_health():
    """Cell health based on ground-truth `actual_anomaly` (dataset label).

    The ML model is a separate concern and shown on the Anomaly Diagnosis page.
    Health here reflects what the network is *really* experiencing.
    """
    conn = get_db()

    cell_rows = conn.execute(
        """SELECT cell_id,
                  COUNT(*) AS total,
                  SUM(actual_anomaly) AS anomaly_count,
                  MAX(actual_anomaly_type) AS dominant_anomaly_type,
                  COUNT(DISTINCT ue_id) AS ue_count
           FROM predictions
           WHERE ingested_at >= datetime('now', '-24 hours')
           GROUP BY cell_id"""
    ).fetchall()

    ue_row = conn.execute(
        """SELECT COUNT(DISTINCT ue_id) AS active_ues,
                  COUNT(*) AS total_records
           FROM predictions
           WHERE ingested_at >= datetime('now', '-24 hours')
             AND ue_id IS NOT NULL"""
    ).fetchone()
    conn.close()

    cells = []
    healthy = degraded = critical = 0

    for r in cell_rows:
        d = dict(r)
        total = d["total"] or 1
        rate = (d["anomaly_count"] or 0) / total
        # Tuned for the dataset's ~5% baseline anomaly rate.
        if rate < 0.02:
            status = "Healthy"
            healthy += 1
        elif rate < 0.07:
            status = "Degraded"
            degraded += 1
        else:
            status = "Critical"
            critical += 1
        d["anomaly_rate"] = round(rate, 4)
        d["status"] = status
        cells.append(d)

    return {
        "total_cells": len(cells),
        "active_ues": ue_row["active_ues"] if ue_row else 0,
        "total_records": ue_row["total_records"] if ue_row else 0,
        "healthy": healthy,
        "degraded": degraded,
        "critical": critical,
        "cells": sorted(cells, key=lambda x: x["anomaly_count"] or 0, reverse=True),
    }


@app.get("/api/anomaly-timeline")
def get_anomaly_timeline(
    hours: int = 24,
    slice_type: Optional[str] = None,
):
    conn = get_db()
    clauses = ["ingested_at >= datetime('now', ?)"]
    params: list = [f"-{hours} hours"]
    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    where = "WHERE " + " AND ".join(clauses)
    rows = conn.execute(
        f"""SELECT strftime('%H:00', ingested_at) AS hour,
                   slice_type,
                   SUM(ml_prediction) AS anomaly_count,
                   COUNT(*) AS total
            FROM predictions {where}
            GROUP BY hour, slice_type
            ORDER BY hour""",
        params,
    ).fetchall()
    conn.close()
    return rows_to_dicts(rows)


@app.get("/api/anomaly-summary")
def get_anomaly_summary(
    slice_type: Optional[str] = None,
    cell_id: Optional[str] = None,
    hours: int = 24,
):
    conn = get_db()
    clauses = ["ingested_at >= datetime('now', ?)"]
    params: list = [f"-{hours} hours"]
    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    if cell_id:
        clauses.append("cell_id = ?")
        params.append(cell_id)
    where = "WHERE " + " AND ".join(clauses)

    kpi_rows = conn.execute(
        f"""SELECT COUNT(*) AS total,
                   SUM(ml_prediction) AS active_anomalies,
                   COUNT(DISTINCT cell_id) AS affected_cells,
                   COUNT(DISTINCT ue_id) AS affected_ues
            FROM predictions {where}""",
        params,
    ).fetchone()
    anom_where = where + " AND ml_prediction = 1"
    type_rows = conn.execute(
        f"""SELECT ml_anomaly_type, COUNT(*) AS count
            FROM predictions {anom_where}
            GROUP BY ml_anomaly_type
            ORDER BY count DESC""",
        params,
    ).fetchall()

    cell_rows = conn.execute(
        f"""SELECT cell_id,
                   COUNT(*) AS total,
                   SUM(ml_prediction) AS anomaly_count
            FROM predictions {where}
            GROUP BY cell_id""",
        params,
    ).fetchall()

    dominant_type_rows = conn.execute(
        f"""SELECT cell_id, ml_anomaly_type, COUNT(*) AS cnt
            FROM predictions {anom_where}
            GROUP BY cell_id, ml_anomaly_type""",
        params,
    ).fetchall()
    dominant_by_cell = {}
    for r in dominant_type_rows:
        d = dict(r)
        cur = dominant_by_cell.get(d["cell_id"])
        if cur is None or d["cnt"] > cur["cnt"]:
            dominant_by_cell[d["cell_id"]] = d

    severity_ranking = []
    for r in cell_rows:
        d = dict(r)
        total = d["total"] or 1
        rate = (d["anomaly_count"] or 0) / total
        if rate >= 0.2:
            severity = "Critical"
        elif rate >= 0.05:
            severity = "Medium"
        else:
            severity = "Low"
        dom = dominant_by_cell.get(d["cell_id"])
        severity_ranking.append({
            "cell_id": d["cell_id"],
            "dominant_anomaly_type": dom["ml_anomaly_type"] if dom else None,
            "anomaly_rate": round(rate, 4),
            "anomaly_count": d["anomaly_count"] or 0,
            "severity": severity,
        })
    severity_ranking.sort(key=lambda x: x["anomaly_count"], reverse=True)

    # Approximate duration: rows are ~1 second apart → count ≈ seconds
    # Return in minutes for display
    active_anomalies = kpi_rows["active_anomalies"] or 0
    avg_anomaly_duration_min = round((active_anomalies / max(len(dominant_by_cell), 1)) / 60.0, 2)

    type_distribution = [dict(r) for r in type_rows]
    dominant_type = type_distribution[0]["ml_anomaly_type"] if type_distribution else None

    conn.close()
    return {
        "active_anomalies": active_anomalies,
        "dominant_anomaly_type": dominant_type,
        "affected_cells": kpi_rows["affected_cells"] or 0,
        "affected_ues": kpi_rows["affected_ues"] or 0,
        "avg_anomaly_duration_min": avg_anomaly_duration_min,
        "type_distribution": type_distribution,
        "severity_ranking": severity_ranking[:20],
    }


@app.get("/api/anomaly-timeline-by-type")
def get_anomaly_timeline_by_type(
    hours: int = 24,
    slice_type: Optional[str] = None,
):
    conn = get_db()
    clauses = ["ingested_at >= datetime('now', ?)", "ml_prediction = 1"]
    params: list = [f"-{hours} hours"]
    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    where = "WHERE " + " AND ".join(clauses)
    rows = conn.execute(
        f"""SELECT strftime('%H:00', ingested_at) AS hour,
                   ml_anomaly_type,
                   COUNT(*) AS count
            FROM predictions {where}
            GROUP BY hour, ml_anomaly_type
            ORDER BY hour""",
        params,
    ).fetchall()
    conn.close()
    return rows_to_dicts(rows)


@app.get("/api/sla-breach-trend")
def get_sla_breach_trend(hours: int = 24):
    """SLA breaches grouped by hour. Returns a 'bucket' string label per row."""
    conn = get_db()
    rows = conn.execute(
        """SELECT strftime('%Y-%m-%d %H:00', created_at) AS bucket,
                  slice_type,
                  COUNT(*) AS breach_count
           FROM alerts
           WHERE created_at >= datetime('now', ?)
           GROUP BY bucket, slice_type
           ORDER BY bucket""",
        (f"-{hours} hours",),
    ).fetchall()
    conn.close()
    return rows_to_dicts(rows)


@app.get("/api/cells")
def list_cells():
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT cell_id FROM predictions WHERE cell_id IS NOT NULL ORDER BY cell_id"
    ).fetchall()
    conn.close()
    return [r["cell_id"] for r in rows]


# ---------------------------------------------------------------------------
# Forecasting (naive linear trend — swap in BiLSTM later)
# ---------------------------------------------------------------------------

FORECASTABLE_KPIS = {
    "one_way_latency_ms":              {"direction": "max", "unit": "ms"},
    "bler_percent":                    {"direction": "max", "unit": "%"},
    "throughput_dl_mbps":              {"direction": "min", "unit": "Mbps"},
    "handover_success_rate_percent":   {"direction": "min", "unit": "%"},
}


@app.get("/api/forecast/{kpi}")
def get_forecast(
    kpi: str,
    cell_id: Optional[str] = None,
    slice_type: Optional[str] = None,
    history_points: int = 30,
    forecast_points: int = 10,
    model: str = "naive",
):
    if kpi not in FORECASTABLE_KPIS:
        return {"error": f"Unsupported KPI. Use one of {list(FORECASTABLE_KPIS)}"}

    conn = get_db()
    clauses, params = [f"{kpi} IS NOT NULL"], []
    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    if cell_id:
        clauses.append("cell_id = ?")
        params.append(cell_id)
    where = "WHERE " + " AND ".join(clauses)
    params.append(history_points)

    rows = conn.execute(
        f"""SELECT timestamp, {kpi} AS value
            FROM predictions {where}
            ORDER BY timestamp DESC LIMIT ?""",
        params,
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "kpi": kpi, "historical": [], "forecast": [],
            "threshold": None, "sla_status": "unknown", "predicted_value": None,
        }

    rows = list(reversed(rows))
    history = [dict(r) for r in rows]
    values = np.array([float(r["value"]) for r in history])

    # Naive linear regression on index → value
    from sklearn.linear_model import LinearRegression
    X = np.arange(len(values)).reshape(-1, 1)
    reg = LinearRegression().fit(X, values)
    future_idx = np.arange(len(values), len(values) + forecast_points).reshape(-1, 1)
    forecast_values = reg.predict(future_idx)

    forecast_list = [
        {"step": int(i + 1), "value": float(v)}
        for i, v in enumerate(forecast_values)
    ]
    predicted_value = float(forecast_values[-1])

    # Infer threshold from SLA table.
    # Priority: explicit slice_type > slice derived from cell_id > worst-case across slices.
    threshold = None
    direction = FORECASTABLE_KPIS[kpi]["direction"]
    effective_slice = slice_type
    if not effective_slice and cell_id:
        cell_slice_row = get_db().execute(
            "SELECT slice_type FROM predictions WHERE cell_id = ? "
            "ORDER BY ingested_at DESC LIMIT 1",
            (cell_id,),
        ).fetchone()
        if cell_slice_row:
            effective_slice = cell_slice_row["slice_type"]
    if effective_slice and effective_slice in SLA_THRESHOLDS:
        threshold = SLA_THRESHOLDS[effective_slice].get(kpi)
    else:
        vals = [t[kpi] for t in SLA_THRESHOLDS.values() if kpi in t]
        threshold = (min(vals) if direction == "max" else max(vals)) if vals else None

    sla_status = "unknown"
    if threshold is not None:
        if direction == "max":
            sla_status = "breach_predicted" if predicted_value > threshold else "within_threshold"
        else:
            sla_status = "breach_predicted" if predicted_value < threshold else "within_threshold"

    return {
        "kpi": kpi,
        "unit": FORECASTABLE_KPIS[kpi]["unit"],
        "historical": history,
        "forecast": forecast_list,
        "threshold": threshold,
        "predicted_value": round(predicted_value, 3),
        "sla_status": sla_status,
        "model": model,
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    manager.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.remove(websocket)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
