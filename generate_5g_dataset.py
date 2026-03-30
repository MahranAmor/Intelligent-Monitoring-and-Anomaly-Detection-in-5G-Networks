"""
Générateur de dataset synthétique 5G — version corrigée pour la prédiction.
Projet PFA - Ayoub & Mahran

CORRECTIONS APPLIQUÉES vs. version originale :
  1. Chaque cellule possède sa PROPRE série temporelle continue.
     Les KPIs évoluent de façon cohérente dans le temps pour chaque cellule.
  2. Corrélation temporelle via processus AR(1) :
     value[t] = alpha * value[t-1] + (1-alpha) * target + bruit
     → les modèles LSTM/GRU ont un vrai signal temporel à apprendre.
  3. Les anomalies sont des ÉVÉNEMENTS TEMPORELS (durée 3-15 pas)
     qui débutent, s'intensifient, puis se résolvent — comme en production.
  4. Chaque cellule a un type de slice FIXE (pas tiré aléatoirement à chaque ligne).

Usage:
    python3 generate_5g_dataset.py
    python3 generate_5g_dataset.py --num-cells 50 --steps-per-cell 6000 --anomaly-ratio 0.05
"""

import argparse
import csv
import os
import random
from datetime import datetime, timedelta

import numpy as np

# ============================================================================
# Configuration KPI (inchangée)
# ============================================================================

KPI_CONFIG = {
    "one_way_latency_ms": {
        "kpi_id": "KPI-1", "unit": "ms",
        "ranges": {"eMBB": (1.0, 20.0), "URLLC": (0.1, 5.0), "mMTC": (10.0, 100.0)},
        "alpha": 0.85,   # smoothing factor — higher = slower changes
    },
    "jitter_ms": {
        "kpi_id": "KPI-2", "unit": "ms",
        "ranges": {"eMBB": (1.0, 10.0), "URLLC": (0.01, 1.0), "mMTC": (5.0, 20.0)},
        "alpha": 0.80,
    },
    "rtt_ms": {
        "kpi_id": "KPI-3", "unit": "ms",
        "ranges": {"eMBB": (10.0, 40.0), "URLLC": (0.5, 10.0), "mMTC": (20.0, 200.0)},
        "alpha": 0.85,
    },
    "packet_delay_budget_ms": {
        "kpi_id": "KPI-18", "unit": "ms",
        "ranges": {"eMBB": (50.0, 100.0), "URLLC": (0.5, 1.0), "mMTC": (50.0, 100.0)},
        "alpha": 0.90,
    },
    "handover_interruption_time_ms": {
        "kpi_id": "KPI-16", "unit": "ms",
        "ranges": {"eMBB": (5.0, 50.0), "URLLC": (0.5, 10.0), "mMTC": (10.0, 60.0)},
        "alpha": 0.80,
    },
    "reliability_percent": {
        "kpi_id": "KPI-4", "unit": "%",
        "ranges": {"eMBB": (99.0, 100.0), "URLLC": (99.999, 100.0), "mMTC": (95.0, 100.0)},
        "alpha": 0.92,
    },
    "packet_loss_percent": {
        "kpi_id": "KPI-5", "unit": "%",
        "ranges": {"eMBB": (0.0, 1.0), "URLLC": (0.0, 0.001), "mMTC": (0.0, 5.0)},
        "alpha": 0.78,
    },
    "packet_loss_rate_percent": {
        "kpi_id": "KPI-10", "unit": "%",
        "ranges": {"eMBB": (0.0, 1.0), "URLLC": (0.0, 0.001), "mMTC": (0.0, 5.0)},
        "alpha": 0.78,
    },
    "bler_percent": {
        "kpi_id": "KPI-14", "unit": "%",
        "ranges": {"eMBB": (0.0, 10.0), "URLLC": (0.0, 1.0), "mMTC": (0.0, 15.0)},
        "alpha": 0.80,
    },
    "throughput_dl_mbps": {
        "kpi_id": "KPI-6", "unit": "Mbps",
        "ranges": {"eMBB": (100.0, 20000.0), "URLLC": (10.0, 200.0), "mMTC": (0.01, 1.0)},
        "alpha": 0.88,
    },
    "throughput_ul_mbps": {
        "kpi_id": "KPI-7", "unit": "Mbps",
        "ranges": {"eMBB": (50.0, 10000.0), "URLLC": (10.0, 200.0), "mMTC": (0.01, 1.0)},
        "alpha": 0.88,
    },
    "spectral_efficiency_bps_hz": {
        "kpi_id": "KPI-9", "unit": "bits/s/Hz",
        "ranges": {"eMBB": (10.0, 30.0), "URLLC": (5.0, 15.0), "mMTC": (1.0, 5.0)},
        "alpha": 0.87,
    },
    "handover_success_rate_percent": {
        "kpi_id": "KPI-15", "unit": "%",
        "ranges": {"eMBB": (99.0, 100.0), "URLLC": (99.0, 100.0), "mMTC": (99.0, 100.0)},
        "alpha": 0.90,
    },
    "energy_efficiency_bits_per_joule": {
        "kpi_id": "KPI-17", "unit": "bits/J",
        "ranges": {"eMBB": (1e6, 1e8), "URLLC": (1e7, 1e9), "mMTC": (1e4, 1e6)},
        "alpha": 0.88,
    },
}

KPI_COLUMNS = list(KPI_CONFIG.keys())

# ============================================================================
# Types d'anomalies
# ============================================================================

ANOMALY_TYPES = [
    "network_congestion",
    "interference",
    "hardware_failure",
    "handover_failure",
    "signal_degradation",
    "security_attack",
    "backhaul_issue",
    "overload",
]

SLICE_TYPES   = ["eMBB", "URLLC", "mMTC"]
SLICE_WEIGHTS = [0.50, 0.30, 0.20]

GEO_LAT_RANGE = (33.5, 34.1)
GEO_LON_RANGE = (-7.7, -7.4)


# ============================================================================
# Helpers
# ============================================================================

def _noise_std(low, high):
    """Small noise proportional to the KPI range."""
    return (high - low) * 0.015


def _clamp(value, low, high):
    return max(low, min(high, value))


def _round_kpi(value, kpi_conf):
    unit = kpi_conf["unit"]
    if unit == "bits/J":
        return round(value, 0)
    elif unit == "%" and max(kpi_conf["ranges"].values(), key=lambda r: r[1])[1] <= 1.0:
        return round(value, 6)
    else:
        return round(value, 4)


# ============================================================================
# Anomaly multipliers applied to a running KPI value
# ============================================================================

def _anomaly_multipliers(anomaly_type):
    """
    Returns a dict of {kpi_name: multiplier} for a given anomaly type.
    Multipliers > 1 increase the KPI, < 1 decrease it.
    Applied progressively over the anomaly window.
    """
    m = {}
    if anomaly_type == "network_congestion":
        m["one_way_latency_ms"]       = random.uniform(2.5, 5.0)
        m["rtt_ms"]                   = random.uniform(2.5, 5.0)
        m["jitter_ms"]                = random.uniform(2.0, 4.0)
        m["packet_delay_budget_ms"]   = random.uniform(1.5, 3.0)
        m["throughput_dl_mbps"]       = random.uniform(0.1, 0.4)
        m["throughput_ul_mbps"]       = random.uniform(0.1, 0.4)
        m["packet_loss_percent"]      = random.uniform(5.0, 20.0)
        m["packet_loss_rate_percent"] = random.uniform(5.0, 20.0)

    elif anomaly_type == "interference":
        m["bler_percent"]                  = random.uniform(4.0, 8.0)
        m["spectral_efficiency_bps_hz"]    = random.uniform(0.2, 0.5)
        m["throughput_dl_mbps"]            = random.uniform(0.2, 0.5)
        m["throughput_ul_mbps"]            = random.uniform(0.3, 0.6)
        m["jitter_ms"]                     = random.uniform(1.5, 3.0)
        m["packet_loss_percent"]           = random.uniform(3.0, 10.0)
        m["packet_loss_rate_percent"]      = random.uniform(3.0, 10.0)

    elif anomaly_type == "hardware_failure":
        m["reliability_percent"]           = random.uniform(0.85, 0.95)
        m["throughput_dl_mbps"]            = random.uniform(0.05, 0.2)
        m["throughput_ul_mbps"]            = random.uniform(0.05, 0.2)
        m["one_way_latency_ms"]            = random.uniform(5.0, 15.0)
        m["rtt_ms"]                        = random.uniform(5.0, 15.0)
        m["packet_loss_percent"]           = random.uniform(10.0, 30.0)
        m["packet_loss_rate_percent"]      = random.uniform(10.0, 30.0)
        m["bler_percent"]                  = random.uniform(5.0, 10.0)
        m["energy_efficiency_bits_per_joule"] = random.uniform(0.1, 0.3)

    elif anomaly_type == "handover_failure":
        m["handover_success_rate_percent"] = random.uniform(0.6, 0.85)
        m["handover_interruption_time_ms"] = random.uniform(5.0, 20.0)
        m["one_way_latency_ms"]            = random.uniform(2.0, 6.0)
        m["rtt_ms"]                        = random.uniform(2.0, 6.0)
        m["packet_loss_percent"]           = random.uniform(3.0, 15.0)
        m["packet_loss_rate_percent"]      = random.uniform(3.0, 15.0)

    elif anomaly_type == "signal_degradation":
        m["spectral_efficiency_bps_hz"]    = random.uniform(0.3, 0.6)
        m["throughput_dl_mbps"]            = random.uniform(0.3, 0.6)
        m["throughput_ul_mbps"]            = random.uniform(0.3, 0.6)
        m["bler_percent"]                  = random.uniform(2.0, 5.0)
        m["reliability_percent"]           = random.uniform(0.93, 0.98)
        m["jitter_ms"]                     = random.uniform(1.5, 3.0)

    elif anomaly_type == "security_attack":
        m["one_way_latency_ms"]            = random.uniform(10.0, 50.0)
        m["rtt_ms"]                        = random.uniform(10.0, 50.0)
        m["jitter_ms"]                     = random.uniform(5.0, 20.0)
        m["packet_loss_percent"]           = random.uniform(20.0, 50.0)
        m["packet_loss_rate_percent"]      = random.uniform(20.0, 50.0)
        m["throughput_dl_mbps"]            = random.uniform(0.01, 0.1)
        m["throughput_ul_mbps"]            = random.uniform(0.01, 0.1)
        m["reliability_percent"]           = random.uniform(0.70, 0.90)

    elif anomaly_type == "backhaul_issue":
        m["one_way_latency_ms"]            = random.uniform(3.0, 10.0)
        m["rtt_ms"]                        = random.uniform(3.0, 10.0)
        m["packet_delay_budget_ms"]        = random.uniform(2.0, 5.0)
        m["throughput_dl_mbps"]            = random.uniform(0.1, 0.3)
        m["throughput_ul_mbps"]            = random.uniform(0.1, 0.3)
        m["jitter_ms"]                     = random.uniform(3.0, 8.0)

    elif anomaly_type == "overload":
        m["throughput_dl_mbps"]            = random.uniform(0.2, 0.5)
        m["throughput_ul_mbps"]            = random.uniform(0.2, 0.5)
        m["one_way_latency_ms"]            = random.uniform(2.0, 4.0)
        m["rtt_ms"]                        = random.uniform(2.0, 4.0)
        m["packet_loss_percent"]           = random.uniform(1.0, 8.0)
        m["packet_loss_rate_percent"]      = random.uniform(1.0, 8.0)
        m["bler_percent"]                  = random.uniform(2.0, 4.0)
        m["reliability_percent"]           = random.uniform(0.95, 0.99)

    return m


# ============================================================================
# Per-cell time series generator
# ============================================================================

def generate_cell_series(slice_type, timestamps):
    """
    Generates a full time series for a single cell.

    Strategy:
    - KPI values evolve via AR(1): v[t] = alpha * v[t-1] + (1-alpha) * mean + noise
    - Anomaly events are temporal windows (3–15 timesteps) where multipliers
      are applied smoothly (ramp-up / ramp-down) to simulate realistic network events.
    - Each cell gets a FIXED slice_type for the entire series.
    """
    T = len(timestamps)

    # ── 1. Initialize KPI state at the midpoint of normal range ──────────────
    state = {}
    for kpi, conf in KPI_CONFIG.items():
        low, high = conf["ranges"][slice_type]
        state[kpi] = low + (high - low) * 0.5  # start at midpoint

    # ── 2. Schedule anomaly windows ──────────────────────────────────────────
    #   Each anomaly: (start_step, duration, anomaly_type)
    #   Windows never overlap. Min gap between events = 10 steps.
    anomaly_schedule = []   # list of (start, end, atype)
    occupied = np.zeros(T, dtype=bool)

    # Target ~5% anomaly coverage across the time series
    target_anomaly_steps = int(T * 0.05)
    attempts = 0
    total_scheduled = 0

    while total_scheduled < target_anomaly_steps and attempts < 500:
        duration   = random.randint(3, 15)
        start      = random.randint(10, T - duration - 10)
        end        = start + duration
        gap_start  = max(0, start - 10)
        gap_end    = min(T, end + 10)

        if not occupied[gap_start:gap_end].any():
            atype = random.choice(ANOMALY_TYPES)
            anomaly_schedule.append((start, end, atype))
            occupied[start:end] = True
            total_scheduled += duration

        attempts += 1

    # Build per-step anomaly label array
    step_anomaly      = ["normal"] * T
    step_is_anomaly   = [0] * T
    for (start, end, atype) in anomaly_schedule:
        for t in range(start, end):
            step_anomaly[t]    = atype
            step_is_anomaly[t] = 1

    # ── 3. Generate the time series step by step ──────────────────────────────
    rows = []

    for t in range(T):
        # Base AR(1) update for each KPI
        new_state = {}
        for kpi, conf in KPI_CONFIG.items():
            low, high   = conf["ranges"][slice_type]
            alpha       = conf["alpha"]
            mean_val    = low + (high - low) * 0.5
            noise       = np.random.normal(0, _noise_std(low, high))
            new_val     = alpha * state[kpi] + (1 - alpha) * mean_val + noise
            new_state[kpi] = _clamp(new_val, low * 0.5, high * 2.0)

        # Apply anomaly multipliers if inside an anomaly window
        atype = step_anomaly[t]
        if atype != "normal":
            # Find which window we're in to compute ramp factor (0→1→0)
            for (start, end, ev_type) in anomaly_schedule:
                if start <= t < end and ev_type == atype:
                    window_len  = end - start
                    pos         = t - start
                    # Smooth ramp: rises in first 30%, peaks, drops in last 30%
                    ramp_frac   = min(pos / max(window_len * 0.3, 1),
                                      (window_len - 1 - pos) / max(window_len * 0.3, 1), 1.0)
                    mults = _anomaly_multipliers(atype)
                    for kpi, mult in mults.items():
                        low, high = KPI_CONFIG[kpi]["ranges"][slice_type]
                        base      = new_state[kpi]
                        # Interpolate between normal and full anomaly value
                        target    = base * mult
                        new_state[kpi] = _clamp(
                            base + ramp_frac * (target - base),
                            low * 0.01, high * 60.0
                        )
                    break

        state = new_state

        # Build row
        ts = timestamps[t]
        row = {
            "timestamp" : ts.strftime("%Y-%m-%d %H:%M:%S"),
            "slice_type": slice_type,
            "latitude"  : round(random.gauss(33.8, 0.005), 6),
            "longitude" : round(random.gauss(-7.55, 0.005), 6),
        }
        for kpi, conf in KPI_CONFIG.items():
            row[kpi] = _round_kpi(state[kpi], conf)

        row["anomaly"]      = step_is_anomaly[t]
        row["anomaly_type"] = atype

        rows.append(row)

    return rows


# ============================================================================
# Main dataset generator
# ============================================================================

def generate_dataset(steps, anomaly_ratio, seed=42, start_date="2024-01-01"):
    """
    Generates a single continuous time series.
    Total rows = steps  (one row every 5 minutes).
    """
    random.seed(seed)
    np.random.seed(seed)

    start_time = datetime.strptime(start_date, "%Y-%m-%d")
    time_step  = timedelta(minutes=5)
    timestamps = [start_time + time_step * t for t in range(steps)]

    slice_type = random.choices(SLICE_TYPES, weights=SLICE_WEIGHTS, k=1)[0]
    print(f"  Slice type : {slice_type}")
    print(f"  Generating {steps:,} timesteps ...")

    dataset        = generate_cell_series(slice_type, timestamps)
    total_anomalies = sum(r["anomaly"] for r in dataset)

    return dataset, total_anomalies


# ============================================================================
# I/O helpers
# ============================================================================

def save_to_csv(dataset, output_path):
    if not dataset:
        print("No records to save.")
        return
    fieldnames = list(dataset[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path}  ({size_mb:.1f} MB)")


def print_summary(dataset, total_anomalies):
    total = len(dataset)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total rows   : {total:,}")
    print(f"  Normal rows  : {total - total_anomalies:,}  ({(total-total_anomalies)/total*100:.1f}%)")
    print(f"  Anomaly rows : {total_anomalies:,}  ({total_anomalies/total*100:.1f}%)")
    print(f"  Slice type   : {dataset[0]['slice_type']}")

    print(f"\n  Anomaly type distribution:")
    ac = {}
    for r in dataset:
        if r["anomaly"] == 1:
            ac[r["anomaly_type"]] = ac.get(r["anomaly_type"], 0) + 1
    for at, cnt in sorted(ac.items(), key=lambda x: -x[1]):
        print(f"    {at:28s}: {cnt:,}  ({cnt/total_anomalies*100:.1f}%)")

    print(f"\n  Columns ({len(dataset[0])}):")
    print(f"    Dimensions : timestamp, slice_type, latitude, longitude")
    print(f"    KPIs ({len(KPI_COLUMNS)}) : {', '.join(KPI_COLUMNS)}")
    print(f"    Labels     : anomaly, anomaly_type")
    print("=" * 60)


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="5G synthetic dataset generator — single time series"
    )
    parser.add_argument("--start-date",    type=str,   default="2024-01-01",
                        help="Start date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--end-date",      type=str,   default="2026-01-01",
                        help="End date YYYY-MM-DD (default: 2026-01-01)")
    parser.add_argument("--anomaly-ratio", type=float, default=0.05,
                        help="Target anomaly fraction (default: 0.05)")
    parser.add_argument("--output",        type=str,   default="5g_timeseries_dataset.csv",
                        help="Output CSV file path")
    parser.add_argument("--seed",          type=int,   default=42)

    args = parser.parse_args()

    start_dt      = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt        = datetime.strptime(args.end_date,   "%Y-%m-%d")
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    steps         = total_minutes // 5
    total_days    = (end_dt - start_dt).days

    print("=" * 60)
    print("5G DATASET GENERATOR — SINGLE TIME SERIES")
    print("=" * 60)
    print(f"  Date range     : {args.start_date}  →  {args.end_date}  ({total_days} days)")
    print(f"  Total steps    : {steps:,}  (5-min intervals)")
    print(f"  Anomaly target : {args.anomaly_ratio*100:.1f}%")
    print(f"  Output         : {args.output}")
    print()

    dataset, total_anomalies = generate_dataset(
        steps, args.anomaly_ratio, args.seed, start_date=args.start_date
    )
    save_to_csv(dataset, args.output)
    print_summary(dataset, total_anomalies)


if __name__ == "__main__":
    main()