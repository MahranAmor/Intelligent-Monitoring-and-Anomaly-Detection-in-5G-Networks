import os
import sys
import json
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_RAW = os.environ.get("KAFKA_TOPIC_RAW", "raw-5g-data")
TOPIC_PREDICTIONS = os.environ.get("KAFKA_TOPIC_PREDICTIONS", "predictions")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/rf_optimized_binary.pkl")
MULTICLASS_MODEL_PATH = os.environ.get("MULTICLASS_MODEL_PATH", "/app/models/rf_optimized_multiclass.pkl")
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH", "/app/models/label_encoder_slice_type.pkl")
LABEL_ENCODER_ANOMALY_TYPE_PATH = os.environ.get("LABEL_ENCODER_ANOMALY_TYPE_PATH", "/app/models/label_encoder_anomaly_type.pkl")

# Exact feature columns both RF models were trained on (16 features)
FEATURE_COLUMNS = [
    "slice_type",
    "latitude",
    "longitude",
    "one_way_latency_ms",
    "jitter_ms",
    "packet_delay_budget_ms",
    "handover_interruption_time_ms",
    "packet_loss_percent",
    "throughput_dl_mbps",
    "handover_success_rate_percent",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
]


def load_models():
    print(f"[Consumer] Loading binary model from {MODEL_PATH}", flush=True)
    with open(MODEL_PATH, "rb") as f:
        binary_model = pickle.load(f)

    print(f"[Consumer] Loading multiclass model from {MULTICLASS_MODEL_PATH}", flush=True)
    with open(MULTICLASS_MODEL_PATH, "rb") as f:
        multiclass_model = pickle.load(f)

    print(f"[Consumer] Loading slice type encoder from {LABEL_ENCODER_PATH}", flush=True)
    le_slice = joblib.load(LABEL_ENCODER_PATH)
    print(f"[Consumer]   slice classes: {list(le_slice.classes_)}", flush=True)

    print(f"[Consumer] Loading anomaly type encoder from {LABEL_ENCODER_ANOMALY_TYPE_PATH}", flush=True)
    le_anomaly_type = joblib.load(LABEL_ENCODER_ANOMALY_TYPE_PATH)
    print(f"[Consumer]   anomaly type classes: {list(le_anomaly_type.classes_)}", flush=True)

    return binary_model, multiclass_model, le_slice, le_anomaly_type


def connect_kafka(retries=30, delay=5):
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                TOPIC_RAW,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                group_id="ml-anomaly-detector",
                auto_offset_reset="earliest",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print(f"[Consumer] Connected to Kafka on attempt {attempt}", flush=True)
            return consumer, producer
        except NoBrokersAvailable:
            print(f"[Consumer] Kafka not ready (attempt {attempt}/{retries}), retrying in {delay}s...", flush=True)
            time.sleep(delay)
    print("[Consumer] Could not connect to Kafka after max retries. Exiting.", flush=True)
    sys.exit(1)


def parse_timestamp(ts_str):
    try:
        ts = pd.to_datetime(ts_str)
        return ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second
    except Exception:
        return 2026, 1, 1, 0, 0, 0


def main():
    print(f"[Consumer] Starting. Topics: {TOPIC_RAW} -> {TOPIC_PREDICTIONS}", flush=True)

    binary_model, multiclass_model, le_slice, le_anomaly_type = load_models()
    consumer, producer = connect_kafka()

    print("[Consumer] Listening for messages...", flush=True)
    count = 0

    for msg in consumer:
        data = msg.value

        # Encode slice_type
        slice_type_str = data.get("slice_type", "eMBB")
        try:
            slice_type_encoded = int(le_slice.transform([slice_type_str])[0])
        except Exception:
            slice_type_encoded = 0

        # Parse timestamp into date parts
        year, month, day, hour, minute, second = parse_timestamp(data.get("timestamp"))

        # Build feature row (shared by both models)
        feature_dict = {
            "slice_type": slice_type_encoded,
            "latitude": data.get("latitude", 0),
            "longitude": data.get("longitude", 0),
            "one_way_latency_ms": data.get("one_way_latency_ms", 0),
            "jitter_ms": data.get("jitter_ms", 0),
            "packet_delay_budget_ms": data.get("packet_delay_budget_ms", 0),
            "handover_interruption_time_ms": data.get("handover_interruption_time_ms", 0),
            "packet_loss_percent": data.get("packet_loss_percent", 0),
            "throughput_dl_mbps": data.get("throughput_dl_mbps", 0),
            "handover_success_rate_percent": data.get("handover_success_rate_percent", 0),
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }

        X = pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)
        X = X.fillna(0)

        # Binary classification: normal (0) vs anomaly (1)
        binary_pred = int(binary_model.predict(X)[0])
        binary_probas = binary_model.predict_proba(X)[0]
        binary_confidence = float(binary_probas[binary_pred])

        # Multiclass classification: predict anomaly type (run on all rows)
        multi_pred_int = int(multiclass_model.predict(X)[0])
        multi_probas = multiclass_model.predict_proba(X)[0]
        multi_confidence = float(multi_probas[multi_pred_int])
        try:
            multi_pred_label = str(le_anomaly_type.inverse_transform([multi_pred_int])[0])
        except Exception:
            multi_pred_label = "normal"

        output = {
            # Identity fields
            "timestamp": data.get("timestamp"),
            "cell_id": data.get("cell_id"),
            "slice_type": data.get("slice_type"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            # Key KPIs
            "one_way_latency_ms": data.get("one_way_latency_ms"),
            "jitter_ms": data.get("jitter_ms"),
            "rtt_ms": data.get("rtt_ms"),
            "throughput_dl_mbps": data.get("throughput_dl_mbps"),
            "throughput_ul_mbps": data.get("throughput_ul_mbps"),
            "packet_loss_percent": data.get("packet_loss_percent"),
            "reliability_percent": data.get("reliability_percent"),
            "bler_percent": data.get("bler_percent"),
            "handover_success_rate_percent": data.get("handover_success_rate_percent"),
            "energy_efficiency_bits_per_joule": data.get("energy_efficiency_bits_per_joule"),
            # Binary ML results
            "ml_prediction": binary_pred,
            "ml_confidence": binary_confidence,
            "ml_anomaly_label": "anomaly" if binary_pred == 1 else "normal",
            # Multiclass ML results
            "ml_anomaly_type": multi_pred_label,
            "ml_anomaly_type_confidence": multi_confidence,
            # Ground truth
            "actual_anomaly": data.get("anomaly"),
            "actual_anomaly_type": data.get("anomaly_type"),
        }

        producer.send(TOPIC_PREDICTIONS, value=output)
        count += 1

        if count % 500 == 0:
            print(f"[Consumer] Processed {count} messages.", flush=True)


if __name__ == "__main__":
    main()
