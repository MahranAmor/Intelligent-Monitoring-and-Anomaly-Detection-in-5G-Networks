# 📡 Kafka Streaming Pipeline — Implementation Blueprint

## ⚠️ CRITICAL: Scope & Safety Rules

### DO NOT TOUCH — Existing Project Files
The following files/folders already exist and must **never** be modified, deleted, renamed, or moved:
- Any `.ipynb` notebook files (forecasting notebooks, ML notebooks, EDA notebooks)
- `dataset_generator*.py` or any data generation scripts
- Any existing `*.csv` files in the project root
- Any existing `README.md` in the project root (if it exists)
- Any existing `requirements.txt` in the project root (if it exists)
- Any LaTeX / thesis report files

### WHAT WAS BUILT
A **self-contained subdirectory** called `streaming/` that contains:
- A Docker Compose stack (Zookeeper + Kafka + 3 app services)
- A Kafka **producer** that reads `test_data.csv` and streams rows as JSON
- A Kafka **consumer** that loads the pre-trained `rf_optimized_binary.pkl` (Random Forest) and runs inference on each row
- A **Streamlit dashboard** that consumes predictions and displays real-time charts
- A `train_model.py` script (optional — pre-trained models already exist in `streaming/models/`)

Everything lives inside `streaming/`. The rest of the project is untouched.

---

## 📁 Directory Structure

```
streaming/
├── docker-compose.yml
├── train_model.py              # Optional — pre-trained model already in models/
├── requirements.txt            # For train_model.py (local execution)
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep               # Place test_data.csv here (not committed)
├── models/                    # Pre-trained models copied from kafka_streaming/PKL_models/
│   ├── rf_optimized_binary.pkl          ← binary anomaly classifier (in use)
│   ├── rf_optimized_multiclass.pkl      ← multiclass anomaly type classifier
│   ├── label_encoder_slice_type.pkl     ← LabelEncoder for slice_type (in use)
│   ├── label_encoder_anomaly_type.pkl
│   ├── feature_columns.pkl
│   ├── one_way_latency_ms_best_model.pt
│   ├── one_way_latency_ms_feature_cols.pkl
│   ├── one_way_latency_ms_scaler_X.pkl
│   ├── one_way_latency_ms_scaler_y.pkl
│   ├── bler_percent_best_model.pt
│   ├── bler_percent_feature_cols.pkl
│   ├── bler_percent_scaler_X.pkl
│   ├── bler_percent_scaler_y.pkl
│   ├── throughput_dl_mbps_best_model.pt
│   ├── throughput_dl_mbps_feature_cols.pkl
│   ├── throughput_dl_mbps_scaler_X.pkl
│   ├── throughput_dl_mbps_scaler_y.pkl
│   ├── handover_success_rate_percent_best_model.pt
│   ├── handover_success_rate_percent_feature_cols.pkl
│   ├── handover_success_rate_percent_scaler_X.pkl
│   └── handover_success_rate_percent_scaler_y.pkl
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
    └── dashboard.py
```

---

## 🔧 docker-compose.yml

5 services. Environment variables:

| Service   | Variable                                | Value                                                        |
|-----------|-----------------------------------------|--------------------------------------------------------------|
| Zookeeper | `ZOOKEEPER_CLIENT_PORT`                 | `2181`                                                       |
| Zookeeper | `ZOOKEEPER_TICK_TIME`                   | `2000`                                                       |
| Kafka     | `KAFKA_BROKER_ID`                       | `1`                                                          |
| Kafka     | `KAFKA_ZOOKEEPER_CONNECT`               | `zookeeper:2181`                                             |
| Kafka     | `KAFKA_ADVERTISED_LISTENERS`            | `PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092`    |
| Kafka     | `KAFKA_LISTENER_SECURITY_PROTOCOL_MAP`  | `PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT`               |
| Kafka     | `KAFKA_INTER_BROKER_LISTENER_NAME`      | `PLAINTEXT`                                                  |
| Kafka     | `KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR`| `1`                                                          |
| Kafka     | `KAFKA_AUTO_CREATE_TOPICS_ENABLE`       | `true`                                                       |
| Producer  | `KAFKA_BOOTSTRAP_SERVERS`               | `kafka:9092`                                                 |
| Producer  | `KAFKA_TOPIC_RAW`                       | `raw-5g-data`                                                |
| Producer  | `CSV_FILE`                              | `/app/data/test_data.csv`                                    |
| Producer  | `STREAM_DELAY`                          | `0.1`                                                        |
| Consumer  | `KAFKA_BOOTSTRAP_SERVERS`               | `kafka:9092`                                                 |
| Consumer  | `KAFKA_TOPIC_RAW`                       | `raw-5g-data`                                                |
| Consumer  | `KAFKA_TOPIC_PREDICTIONS`               | `predictions`                                                |
| Consumer  | `MODEL_PATH`                            | `/app/models/rf_optimized_binary.pkl`                        |
| Consumer  | `MULTICLASS_MODEL_PATH`                 | `/app/models/rf_optimized_multiclass.pkl`                    |
| Consumer  | `LABEL_ENCODER_PATH`                    | `/app/models/label_encoder_slice_type.pkl`                   |
| Consumer  | `LABEL_ENCODER_ANOMALY_TYPE_PATH`       | `/app/models/label_encoder_anomaly_type.pkl`                 |
| Dashboard | `KAFKA_BOOTSTRAP_SERVERS`               | `kafka:9092`                                                 |
| Dashboard | `KAFKA_TOPIC_PREDICTIONS`               | `predictions`                                                |

Volume mounts:
- Producer: `./data:/app/data` (read-only)
- Consumer: `./models:/app/models` (read-only)

Health checks:
- Zookeeper: `echo ruok | nc localhost 2181`, interval 10s, retries 5
- Kafka: `kafka-broker-api-versions --bootstrap-server localhost:9092`, interval 10s, retries 10

Dependency chain: `zookeeper → kafka → producer, consumer, dashboard`

---

## 🔧 producer/producer.py

1. Read env vars: `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_TOPIC_RAW`, `CSV_FILE`, `STREAM_DELAY`
2. Retry Kafka connection up to 30 times (5s between retries)
3. `KafkaProducer` with `value_serializer=lambda v: json.dumps(v).encode("utf-8")`
4. Load CSV with `pd.read_csv(CSV_FILE)`
5. Iterate rows → convert to dict → replace NaN with `None` → `producer.send()` → `time.sleep(STREAM_DELAY)`
6. Log progress every 500 rows, call `producer.flush()` at end

---

## 🔧 consumer/consumer.py

### Model Loading

All models are **raw objects** (not dict bundles):

| File | Load with | Type | Classes |
|------|-----------|------|---------|
| `rf_optimized_binary.pkl` | `pickle` | `RandomForestClassifier` | `[0, 1]` (normal/anomaly) |
| `rf_optimized_multiclass.pkl` | `pickle` | `RandomForestClassifier` | `[0..8]` (anomaly types) |
| `label_encoder_slice_type.pkl` | `joblib` | `LabelEncoder` | `['URLLC', 'eMBB', 'mMTC']` |
| `label_encoder_anomaly_type.pkl` | `joblib` | `LabelEncoder` | `['backhaul_issue', 'handover_failure', 'hardware_failure', 'interference', 'network_congestion', 'normal', 'overload', 'security_attack', 'signal_degradation']` |

```python
with open(MODEL_PATH, "rb") as f:
    binary_model = pickle.load(f)
with open(MULTICLASS_MODEL_PATH, "rb") as f:
    multiclass_model = pickle.load(f)
le_slice = joblib.load(LABEL_ENCODER_PATH)
le_anomaly_type = joblib.load(LABEL_ENCODER_ANOMALY_TYPE_PATH)
```

### Feature Columns (16 total — exact order required)

```python
FEATURE_COLUMNS = [
    "slice_type",                    # LabelEncoder-encoded (int)
    "latitude",
    "longitude",
    "one_way_latency_ms",
    "jitter_ms",
    "packet_delay_budget_ms",
    "handover_interruption_time_ms",
    "packet_loss_percent",
    "throughput_dl_mbps",
    "handover_success_rate_percent",
    "year",                          # parsed from timestamp
    "month",                         # parsed from timestamp
    "day",                           # parsed from timestamp
    "hour",                          # parsed from timestamp
    "minute",                        # parsed from timestamp
    "second",                        # parsed from timestamp
]
```

### Timestamp Parsing

The `timestamp` column must be parsed into 6 date-part features:
```python
def parse_timestamp(ts_str):
    ts = pd.to_datetime(ts_str)
    return ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second
```

### Per-message Logic

1. Encode `slice_type` with `le_slice.transform([slice_type_str])[0]`
2. Parse `timestamp` into `year, month, day, hour, minute, second`
3. Build 16-feature dict → DataFrame → `fillna(0)` (shared by both models)
4. **Binary**: `binary_model.predict(X)` → `binary_model.predict_proba(X)` → `binary_confidence`
5. **Multiclass**: `multiclass_model.predict(X)` → `le_anomaly_type.inverse_transform(...)` → `ml_anomaly_type` label (run on all rows)
6. Publish enriched message to `TOPIC_PREDICTIONS` including:
   - Identity: `timestamp, cell_id, slice_type, latitude, longitude`
   - KPIs: `one_way_latency_ms, jitter_ms, rtt_ms, throughput_dl_mbps, throughput_ul_mbps, packet_loss_percent, reliability_percent, bler_percent, handover_success_rate_percent, energy_efficiency_bits_per_joule`
   - Binary ML: `ml_prediction` (0/1), `ml_confidence` (float), `ml_anomaly_label` ("normal"/"anomaly")
   - Multiclass ML: `ml_anomaly_type` (string label), `ml_anomaly_type_confidence` (float)
   - Ground truth: `actual_anomaly`, `actual_anomaly_type`

---

## 🔧 train_model.py (optional — pre-trained models exist)

This script is **not required** before running the pipeline since `rf_optimized_binary.pkl` already exists in `streaming/models/`. It is kept for reference / re-training purposes only.

If run, it trains a `DecisionTreeClassifier` on `data/test_data.csv` using the 16 feature columns above and saves the bundle to `models/decision_tree_model.pkl`.

---

## 🔧 dashboard/dashboard.py

**Kafka consumer** (`@st.cache_resource`):
- Topic: `TOPIC_PREDICTIONS`, `auto_offset_reset="latest"`, `consumer_timeout_ms=1000`
- Retry 20 times × 3s

**Session state**: `buffer` (deque maxlen=500), `total_processed`, `total_anomalies`, `correct_predictions`

**Sidebar**: auto-refresh toggle (3s default), refresh slider (1–10s), slice type multi-select, latency threshold, packet loss threshold

**Layout**:
1. KPI row (5 metrics): Total Processed, Anomalies Detected, Model Accuracy, Avg Latency, Avg Throughput DL
2. Time series (2 cols): Latency by slice type + threshold line | Throughput DL by slice type
3. Anomaly analysis (3 cols): Donut (normal/anomaly) | Bar (anomaly types, top 8) | Histogram (confidence by label)
4. Per-cell bar chart: top 20 cells by anomaly rate, `RdYlGn_r` colorscale
5. Recent alerts table: last 10 anomaly rows, `st.dataframe(hide_index=True)`

**Auto-refresh** (end of script):
```python
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
```

---

## 📊 Dataset Reference (`test_data.csv`)

| Column | Type | Role |
|--------|------|------|
| `timestamp` | datetime | Record timestamp — parsed into year/month/day/hour/minute/second for features |
| `cell_id` | string | Base station ID (e.g. `gNB-012`) |
| `ue_id` | string | User equipment ID |
| `slice_type` | string | `eMBB`, `URLLC`, or `mMTC` — LabelEncoder encoded as feature |
| `latitude` | float | Cell latitude |
| `longitude` | float | Cell longitude |
| `one_way_latency_ms` | float | KPI — Latency |
| `jitter_ms` | float | KPI — Latency |
| `rtt_ms` | float | KPI — Latency |
| `packet_delay_budget_ms` | float | KPI — Latency |
| `handover_interruption_time_ms` | float | KPI — Mobility |
| `reliability_percent` | float | KPI — Radio Quality |
| `packet_loss_percent` | float | KPI — Radio Quality |
| `packet_loss_rate_percent` | float | KPI — Radio Quality |
| `bler_percent` | float | KPI — Radio Quality |
| `throughput_dl_mbps` | float | KPI — Capacity |
| `throughput_ul_mbps` | float | KPI — Capacity |
| `spectral_efficiency_bps_hz` | float | KPI — Capacity |
| `handover_success_rate_percent` | float | KPI — Mobility |
| `energy_efficiency_bits_per_joule` | float | KPI — Capacity |
| `anomaly` | int | 0 = normal, 1 = anomaly (target) |
| `anomaly_type` | string | `normal`, `backhaul_issue`, `network_congestion`, `signal_degradation`, `security_attack`, `handover_failure`, `interference`, `overload`, `hardware_failure` |

**Total rows**: 89,472 — **Anomaly ratio**: ~5%

---

## 🔗 Service Communication Map

```
┌─────────────┐     JSON rows     ┌──────────────────────┐
│   Producer   │ ──────────────→  │  Kafka Topic:         │
│  (reads CSV) │                  │  "raw-5g-data"        │
└─────────────┘                  └──────────┬───────────┘
                                            │
                                            ▼
                                 ┌──────────────────────┐
                                 │     ML Consumer       │
                                 │  (Random Forest       │
                                 │   Binary Classifier)  │
                                 └──────────┬───────────┘
                                            │
                                            ▼
┌─────────────┐   enriched JSON  ┌──────────────────────┐
│  Streamlit   │ ←────────────── │  Kafka Topic:         │
│  Dashboard   │                  │  "predictions"        │
└─────────────┘                  └──────────────────────┘
```

- Kafka reachable at `kafka:9092` inside Docker, `localhost:29092` from host
- Dashboard exposed at `localhost:8501`

---

## 🚀 Setup & Run

### Step 1 — Copy the dataset
```bash
cp ../Data/test_data.csv streaming/data/test_data.csv
```

### Step 2 — Start the full stack
No local training needed — models are pre-trained and already in `streaming/models/`.
```bash
cd streaming/
docker-compose up --build
```

### Step 3 — Open the dashboard
Navigate to [http://localhost:8501](http://localhost:8501)

---

## ✅ Validation Checklist

- [ ] `streaming/` directory exists with all files listed above
- [ ] No files outside `streaming/` were modified
- [ ] `docker-compose.yml` defines exactly 5 services: `zookeeper`, `kafka`, `producer`, `consumer`, `dashboard`
- [ ] Each of `producer/`, `consumer/`, `dashboard/` has: `Dockerfile`, `requirements.txt`, and main `.py` script
- [ ] Consumer loads `rf_optimized_binary.pkl` and `rf_optimized_multiclass.pkl` with `pickle`
- [ ] Consumer loads `label_encoder_slice_type.pkl` and `label_encoder_anomaly_type.pkl` with `joblib`
- [ ] Consumer env vars include `MODEL_PATH`, `MULTICLASS_MODEL_PATH`, `LABEL_ENCODER_PATH`, `LABEL_ENCODER_ANOMALY_TYPE_PATH`
- [ ] Consumer output includes both binary fields (`ml_prediction`, `ml_confidence`, `ml_anomaly_label`) and multiclass fields (`ml_anomaly_type`, `ml_anomaly_type_confidence`)
- [ ] Consumer feature columns: exactly 16 (10 KPIs + `slice_type` + `latitude` + `longitude` + `year` + `month` + `day` + `hour` + `minute` + `second`)
- [ ] All Python scripts use `os.environ.get()` for configuration
- [ ] All Kafka connection functions have retry logic (30 retries, 5s delay)
- [ ] Dashboard uses `@st.cache_resource` for Kafka consumer
- [ ] Dashboard uses `st.session_state` for rolling buffer
- [ ] Dashboard has auto-refresh via `time.sleep()` + `st.rerun()`
- [ ] `.gitignore` excludes `*.pkl`, `*.pt`, `data/*.csv`, `__pycache__/`, `venv/`

---

## 🚫 Common Pitfalls to Avoid

1. **Do NOT use `localhost` inside Docker containers** — use service names (`kafka`, `zookeeper`)
2. **Do NOT forget the `-u` flag** in `CMD ["python", "-u", "producer.py"]`
3. **Do NOT skip Kafka health checks** — producers/consumers will crash if Kafka isn't ready
4. **Do NOT use `consumer_timeout_ms` in the consumer service** — only the dashboard needs it
5. **Do NOT load `rf_optimized_binary.pkl` with `joblib`** — it must be loaded with `pickle`
6. **Do NOT load `label_encoder_slice_type.pkl` with `pickle`** — it must be loaded with `joblib`
7. **Do NOT omit timestamp parsing** — `year/month/day/hour/minute/second` are required features
8. **Do NOT commit `*.pkl`, `*.pt`, or `data/*.csv`** to Git
9. **Do NOT modify any notebook files** — this streaming module is completely independent
