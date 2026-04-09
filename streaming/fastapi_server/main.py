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
    conn.commit()
    conn.close()
    print("[FastAPI] Database initialised.", flush=True)


def insert_prediction(conn: sqlite3.Connection, record: dict):
    conn.execute("""
        INSERT INTO predictions (
            timestamp, cell_id, slice_type, latitude, longitude,
            one_way_latency_ms, jitter_ms, rtt_ms,
            throughput_dl_mbps, throughput_ul_mbps,
            packet_loss_percent, reliability_percent, bler_percent,
            handover_success_rate_percent, energy_efficiency_bits_per_joule,
            ml_prediction, ml_confidence, ml_anomaly_label,
            ml_anomaly_type, ml_anomaly_type_confidence,
            actual_anomaly, actual_anomaly_type
        ) VALUES (
            :timestamp, :cell_id, :slice_type, :latitude, :longitude,
            :one_way_latency_ms, :jitter_ms, :rtt_ms,
            :throughput_dl_mbps, :throughput_ul_mbps,
            :packet_loss_percent, :reliability_percent, :bler_percent,
            :handover_success_rate_percent, :energy_efficiency_bits_per_joule,
            :ml_prediction, :ml_confidence, :ml_anomaly_label,
            :ml_anomaly_type, :ml_anomaly_type_confidence,
            :actual_anomaly, :actual_anomaly_type
        )
    """, {k: record.get(k) for k in [
        "timestamp", "cell_id", "slice_type", "latitude", "longitude",
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
        clauses.append("ml_prediction = 1")
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
        f"SELECT * FROM alerts {where} ORDER BY alert_time DESC LIMIT ? OFFSET ?",
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
    conn = get_db()

    # Per-cell anomaly counts for the last 24h
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    cell_rows = conn.execute(
        """SELECT cell_id,
                  COUNT(*) AS total,
                  SUM(ml_prediction) AS anomaly_count,
                  MAX(ml_anomaly_type) AS dominant_anomaly_type
           FROM predictions
           WHERE timestamp >= ?
           GROUP BY cell_id""",
        (cutoff,),
    ).fetchall()
    conn.close()

    cells = []
    healthy = degraded = critical = 0

    for r in cell_rows:
        d = dict(r)
        total = d["total"] or 1
        rate = (d["anomaly_count"] or 0) / total
        if rate == 0:
            status = "Healthy"
            healthy += 1
        elif rate < 0.2:
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
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    clauses = ["timestamp >= ?"]
    params: list = [cutoff]
    if slice_type:
        clauses.append("slice_type = ?")
        params.append(slice_type)
    where = "WHERE " + " AND ".join(clauses)
    rows = conn.execute(
        f"""SELECT strftime('%H:00', timestamp) AS hour,
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
