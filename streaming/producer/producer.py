import os
import sys
import json
import time
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_RAW = os.environ.get("KAFKA_TOPIC_RAW", "raw-5g-data")
CSV_FILE = os.environ.get("CSV_FILE", "/app/data/test_data.csv")
STREAM_DELAY = float(os.environ.get("STREAM_DELAY", "0.1"))


def connect_producer(retries=30, delay=5):
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print(f"[Producer] Connected to Kafka on attempt {attempt}", flush=True)
            return producer
        except NoBrokersAvailable:
            print(f"[Producer] Kafka not ready (attempt {attempt}/{retries}), retrying in {delay}s...", flush=True)
            time.sleep(delay)
    print("[Producer] Could not connect to Kafka after max retries. Exiting.", flush=True)
    sys.exit(1)


def main():
    print(f"[Producer] Starting. CSV={CSV_FILE}, Topic={TOPIC_RAW}, Delay={STREAM_DELAY}s", flush=True)

    producer = connect_producer()

    print(f"[Producer] Loading CSV from {CSV_FILE}", flush=True)
    df = pd.read_csv(CSV_FILE)
    print(f"[Producer] Loaded {len(df)} rows. Streaming...", flush=True)

    for idx, (_, row) in enumerate(df.iterrows()):
        message = row.to_dict()
        message = {k: (None if pd.isna(v) else v) for k, v in message.items()}
        producer.send(TOPIC_RAW, value=message)
        time.sleep(STREAM_DELAY)

        if (idx + 1) % 500 == 0:
            print(f"[Producer] Sent {idx + 1} rows...", flush=True)

    producer.flush()
    print(f"[Producer] Done. Streamed {len(df)} rows to topic '{TOPIC_RAW}'.", flush=True)


if __name__ == "__main__":
    main()
