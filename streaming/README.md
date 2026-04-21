# 5G Network Anomaly Streaming Pipeline

Real-time anomaly detection for 5G networks using Kafka, a Decision Tree classifier, and a Streamlit dashboard.

## Architecture

```
┌─────────────┐     JSON rows     ┌──────────────────────┐
│   Producer   │ ──────────────→  │  Kafka Topic:         │
│  (reads CSV) │                  │  "raw-5g-data"        │
└─────────────┘                  └──────────┬───────────┘
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │     ML Consumer       │
                                 │  (Decision Tree)      │
                                 └──────────┬───────────┘
                                            │
                                            ▼
┌─────────────┐   enriched JSON  ┌──────────────────────┐
│  Streamlit   │ ←────────────── │  Kafka Topic:         │
│  Dashboard   │                  │  "predictions"        │
└─────────────┘                  └──────────────────────┘
```

Docker services: `zookeeper` → `kafka` → `producer`, `consumer`, `dashboard`

## Prerequisites

- Docker Desktop (running)
- Python 3.10+ (for local model training)

## Project Structure

```
streaming/
├── docker-compose.yml
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/               ← place test_data.csv here
├── models/             ← trained model saved here
├── producer/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── producer.py
├── consumer/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── consumer.py
└── dashboard/
    ├── Dockerfile
    ├── requirements.txt
    ├── Home.py                  ← landing page
    ├── .streamlit/config.toml   ← theme
    ├── utils/                   ← FastAPI client + shared CSS
    └── pages/
        ├── 1_Network_Health.py
        ├── 2_Anomaly_Diagnosis.py
        ├── 3_KPI_Forecasting.py
        ├── 4_SLA_Monitoring.py
        └── 5_Live_Stream.py     ← legacy direct-Kafka view
```

## Setup & Run

### Step 1 — Copy the dataset

```bash
cp ../Data/test_data.csv data/test_data.csv
# or wherever your test_data.csv lives in the project
```

### Step 2 — Train the model locally

```bash
cd streaming/
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

This produces `models/decision_tree_model.pkl`.

### Step 3 — Start the full stack

```bash
docker-compose up --build
```

### Step 4 — Open the dashboard

Navigate to [http://localhost:8501](http://localhost:8501).

---

## Useful Docker Commands

```bash
# View logs for a specific service
docker-compose logs -f producer
docker-compose logs -f consumer
docker-compose logs -f dashboard

# Stop everything
docker-compose down

# Rebuild a single service after code change
docker-compose up --build consumer

# Remove volumes too (full reset)
docker-compose down -v
```

## Adjusting Stream Speed

Set `STREAM_DELAY` in `docker-compose.yml` under the `producer` service:

- `"0.1"` → ~10 rows/second (default)
- `"0.5"` → ~2 rows/second (slower, easier to read logs)
- `"0.01"` → ~100 rows/second (stress test)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `NoBrokersAvailable` in logs | Kafka is still starting; services retry automatically (30×5s) |
| Dashboard shows "Waiting for data..." | Consumer may not have started yet; check `docker-compose logs consumer` |
| Model file not found in consumer | Ensure you ran `train_model.py` before `docker-compose up` |
| Port 8501 already in use | Stop any other Streamlit instance or change the port mapping in `docker-compose.yml` |
| Port 29092 already in use | Change the host port in `docker-compose.yml` (`"29093:29092"`) |
