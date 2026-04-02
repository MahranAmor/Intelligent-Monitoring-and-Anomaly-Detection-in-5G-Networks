"""
Générateur de dataset Dashboard 5G — séries temporelles par cellule
Projet PFA - Ayoub & Mahran

Architecture :
  • N cellules gNB, chacune avec son propre type de slice FIXE.
  • Chaque cellule génère une série temporelle AR(1) continue indépendante.
  • Chaque cellule possède K UEs qui suivent la série de la cellule
    avec un bruit individuel léger (variations UE-level).
  • Les anomalies sont injectées au niveau de la cellule → tous les UEs
    de cette cellule sont affectés simultanément.

Colonnes de sortie :
  timestamp | cell_id | ue_id | slice_type | latitude | longitude |
  <14 KPIs> | anomaly | anomaly_type

Usage :
    python3 generate_dashboard_dataset.py
    python3 generate_dashboard_dataset.py --num-cells 30 --ues-per-cell 5 \\
        --end-date 2026-04-01 --output Data/Dashboard_data.csv
"""

import argparse
import csv
import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np

# ============================================================================
# KPI configuration (identique au générateur original)
# ============================================================================

KPI_CONFIG = {
    "one_way_latency_ms": {
        "kpi_id": "KPI-1", "unit": "ms",
        "ranges": {"eMBB": (1.0, 20.0), "URLLC": (0.1, 5.0), "mMTC": (10.0, 100.0)},
        "alpha": 0.85,
        "ue_noise_frac": 0.02,   # UE-level noise as fraction of KPI range
    },
    "jitter_ms": {
        "kpi_id": "KPI-2", "unit": "ms",
        "ranges": {"eMBB": (1.0, 10.0), "URLLC": (0.01, 1.0), "mMTC": (5.0, 20.0)},
        "alpha": 0.80,
        "ue_noise_frac": 0.03,
    },
    "rtt_ms": {
        "kpi_id": "KPI-3", "unit": "ms",
        "ranges": {"eMBB": (10.0, 40.0), "URLLC": (0.5, 10.0), "mMTC": (20.0, 200.0)},
        "alpha": 0.85,
        "ue_noise_frac": 0.02,
    },
    "packet_delay_budget_ms": {
        "kpi_id": "KPI-18", "unit": "ms",
        "ranges": {"eMBB": (50.0, 100.0), "URLLC": (0.5, 1.0), "mMTC": (50.0, 100.0)},
        "alpha": 0.90,
        "ue_noise_frac": 0.01,
    },
    "handover_interruption_time_ms": {
        "kpi_id": "KPI-16", "unit": "ms",
        "ranges": {"eMBB": (5.0, 50.0), "URLLC": (0.5, 10.0), "mMTC": (10.0, 60.0)},
        "alpha": 0.80,
        "ue_noise_frac": 0.05,
    },
    "reliability_percent": {
        "kpi_id": "KPI-4", "unit": "%",
        "ranges": {"eMBB": (99.0, 100.0), "URLLC": (99.999, 100.0), "mMTC": (95.0, 100.0)},
        "alpha": 0.92,
        "ue_noise_frac": 0.005,
    },
    "packet_loss_percent": {
        "kpi_id": "KPI-5", "unit": "%",
        "ranges": {"eMBB": (0.0, 1.0), "URLLC": (0.0, 0.001), "mMTC": (0.0, 5.0)},
        "alpha": 0.78,
        "ue_noise_frac": 0.03,
    },
    "packet_loss_rate_percent": {
        "kpi_id": "KPI-10", "unit": "%",
        "ranges": {"eMBB": (0.0, 1.0), "URLLC": (0.0, 0.001), "mMTC": (0.0, 5.0)},
        "alpha": 0.78,
        "ue_noise_frac": 0.03,
    },
    "bler_percent": {
        "kpi_id": "KPI-14", "unit": "%",
        "ranges": {"eMBB": (0.0, 10.0), "URLLC": (0.0, 1.0), "mMTC": (0.0, 15.0)},
        "alpha": 0.80,
        "ue_noise_frac": 0.03,
    },
    "throughput_dl_mbps": {
        "kpi_id": "KPI-6", "unit": "Mbps",
        "ranges": {"eMBB": (100.0, 20000.0), "URLLC": (10.0, 200.0), "mMTC": (0.01, 1.0)},
        "alpha": 0.88,
        "ue_noise_frac": 0.05,
    },
    "throughput_ul_mbps": {
        "kpi_id": "KPI-7", "unit": "Mbps",
        "ranges": {"eMBB": (50.0, 10000.0), "URLLC": (10.0, 200.0), "mMTC": (0.01, 1.0)},
        "alpha": 0.88,
        "ue_noise_frac": 0.05,
    },
    "spectral_efficiency_bps_hz": {
        "kpi_id": "KPI-9", "unit": "bits/s/Hz",
        "ranges": {"eMBB": (10.0, 30.0), "URLLC": (5.0, 15.0), "mMTC": (1.0, 5.0)},
        "alpha": 0.87,
        "ue_noise_frac": 0.02,
    },
    "handover_success_rate_percent": {
        "kpi_id": "KPI-15", "unit": "%",
        "ranges": {"eMBB": (99.0, 100.0), "URLLC": (99.0, 100.0), "mMTC": (99.0, 100.0)},
        "alpha": 0.90,
        "ue_noise_frac": 0.005,
    },
    "energy_efficiency_bits_per_joule": {
        "kpi_id": "KPI-17", "unit": "bits/J",
        "ranges": {"eMBB": (1e6, 1e8), "URLLC": (1e7, 1e9), "mMTC": (1e4, 1e6)},
        "alpha": 0.88,
        "ue_noise_frac": 0.04,
    },
}

KPI_COLUMNS = list(KPI_CONFIG.keys())

# ============================================================================
# Anomaly types
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

# Casablanca-area geo boundaries
GEO_LAT_RANGE = (33.5, 34.1)
GEO_LON_RANGE = (-7.7, -7.4)


# ============================================================================
# Helpers
# ============================================================================

def _noise_std(low, high):
    return (high - low) * 0.015


def _clamp(value, low, high):
    return max(low, min(high, value))


def _round_kpi(value, conf):
    unit = conf["unit"]
    if unit == "bits/J":
        return round(value, 0)
    elif unit == "%" and max(conf["ranges"].values(), key=lambda r: r[1])[1] <= 1.0:
        return round(value, 6)
    else:
        return round(value, 4)


def _anomaly_multipliers(anomaly_type):
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
        m["reliability_percent"]              = random.uniform(0.85, 0.95)
        m["throughput_dl_mbps"]               = random.uniform(0.05, 0.2)
        m["throughput_ul_mbps"]               = random.uniform(0.05, 0.2)
        m["one_way_latency_ms"]               = random.uniform(5.0, 15.0)
        m["rtt_ms"]                           = random.uniform(5.0, 15.0)
        m["packet_loss_percent"]              = random.uniform(10.0, 30.0)
        m["packet_loss_rate_percent"]         = random.uniform(10.0, 30.0)
        m["bler_percent"]                     = random.uniform(5.0, 10.0)
        m["energy_efficiency_bits_per_joule"] = random.uniform(0.1, 0.3)
    elif anomaly_type == "handover_failure":
        m["handover_success_rate_percent"]    = random.uniform(0.6, 0.85)
        m["handover_interruption_time_ms"]    = random.uniform(5.0, 20.0)
        m["one_way_latency_ms"]               = random.uniform(2.0, 6.0)
        m["rtt_ms"]                           = random.uniform(2.0, 6.0)
        m["packet_loss_percent"]              = random.uniform(3.0, 15.0)
        m["packet_loss_rate_percent"]         = random.uniform(3.0, 15.0)
    elif anomaly_type == "signal_degradation":
        m["spectral_efficiency_bps_hz"]       = random.uniform(0.3, 0.6)
        m["throughput_dl_mbps"]               = random.uniform(0.3, 0.6)
        m["throughput_ul_mbps"]               = random.uniform(0.3, 0.6)
        m["bler_percent"]                     = random.uniform(2.0, 5.0)
        m["reliability_percent"]              = random.uniform(0.93, 0.98)
        m["jitter_ms"]                        = random.uniform(1.5, 3.0)
    elif anomaly_type == "security_attack":
        m["one_way_latency_ms"]               = random.uniform(10.0, 50.0)
        m["rtt_ms"]                           = random.uniform(10.0, 50.0)
        m["jitter_ms"]                        = random.uniform(5.0, 20.0)
        m["packet_loss_percent"]              = random.uniform(20.0, 50.0)
        m["packet_loss_rate_percent"]         = random.uniform(20.0, 50.0)
        m["throughput_dl_mbps"]               = random.uniform(0.01, 0.1)
        m["throughput_ul_mbps"]               = random.uniform(0.01, 0.1)
        m["reliability_percent"]              = random.uniform(0.70, 0.90)
    elif anomaly_type == "backhaul_issue":
        m["one_way_latency_ms"]               = random.uniform(3.0, 10.0)
        m["rtt_ms"]                           = random.uniform(3.0, 10.0)
        m["packet_delay_budget_ms"]           = random.uniform(2.0, 5.0)
        m["throughput_dl_mbps"]               = random.uniform(0.1, 0.3)
        m["throughput_ul_mbps"]               = random.uniform(0.1, 0.3)
        m["jitter_ms"]                        = random.uniform(3.0, 8.0)
    elif anomaly_type == "overload":
        m["throughput_dl_mbps"]               = random.uniform(0.2, 0.5)
        m["throughput_ul_mbps"]               = random.uniform(0.2, 0.5)
        m["one_way_latency_ms"]               = random.uniform(2.0, 4.0)
        m["rtt_ms"]                           = random.uniform(2.0, 4.0)
        m["packet_loss_percent"]              = random.uniform(1.0, 8.0)
        m["packet_loss_rate_percent"]         = random.uniform(1.0, 8.0)
        m["bler_percent"]                     = random.uniform(2.0, 4.0)
        m["reliability_percent"]              = random.uniform(0.95, 0.99)
    return m


# ============================================================================
# Per-cell time series generator
# ============================================================================

def generate_cell_series(cell_id, slice_type, lat, lon, ue_ids, timestamps):
    """
    Generate all rows for a single gNB cell and its UEs.

    Strategy:
    - Cell-level KPI follows AR(1) (same as original generator).
    - Anomaly events are cell-level: all UEs in the cell share the same
      anomaly label but get slightly different KPI values (UE noise).
    - Each UE has a persistent offset drawn at init to simulate different
      UE positions / signal conditions within the cell.
    """
    T       = len(timestamps)
    n_ues   = len(ue_ids)

    # ── Cell-level KPI initial state ─────────────────────────────────────────
    cell_state = {}
    for kpi, conf in KPI_CONFIG.items():
        low, high = conf["ranges"][slice_type]
        cell_state[kpi] = low + (high - low) * 0.5

    # ── UE persistent offsets (drawn once per UE) ─────────────────────────────
    # Each UE has a small multiplier around 1.0 for each KPI
    ue_offsets = []
    for _ in range(n_ues):
        offsets = {}
        for kpi, conf in KPI_CONFIG.items():
            low, high = conf["ranges"][slice_type]
            noise_std = (high - low) * conf["ue_noise_frac"]
            offsets[kpi] = np.random.normal(0.0, noise_std)
        ue_offsets.append(offsets)

    # ── Anomaly schedule (cell-level) ────────────────────────────────────────
    anomaly_schedule = []
    occupied = np.zeros(T, dtype=bool)
    target_anomaly_steps = int(T * 0.05)
    total_scheduled = 0
    attempts = 0

    while total_scheduled < target_anomaly_steps and attempts < 500:
        duration  = random.randint(3, 15)
        start     = random.randint(10, max(11, T - duration - 10))
        end       = start + duration
        gap_start = max(0, start - 10)
        gap_end   = min(T, end + 10)
        if not occupied[gap_start:gap_end].any():
            atype = random.choice(ANOMALY_TYPES)
            anomaly_schedule.append((start, end, atype))
            occupied[start:end] = True
            total_scheduled += duration
        attempts += 1

    step_anomaly    = ["normal"] * T
    step_is_anomaly = [0] * T
    for (start, end, atype) in anomaly_schedule:
        for t in range(start, end):
            step_anomaly[t]    = atype
            step_is_anomaly[t] = 1

    # ── Generate step-by-step ─────────────────────────────────────────────────
    rows = []

    for t in range(T):
        # Cell-level AR(1) update
        new_cell = {}
        for kpi, conf in KPI_CONFIG.items():
            low, high = conf["ranges"][slice_type]
            alpha     = conf["alpha"]
            mean_val  = low + (high - low) * 0.5
            noise     = np.random.normal(0, _noise_std(low, high))
            new_val   = alpha * cell_state[kpi] + (1 - alpha) * mean_val + noise
            new_cell[kpi] = _clamp(new_val, low * 0.5, high * 2.0)

        # Apply anomaly multipliers (cell-level)
        atype = step_anomaly[t]
        if atype != "normal":
            for (start, end, ev_type) in anomaly_schedule:
                if start <= t < end and ev_type == atype:
                    window_len = end - start
                    pos        = t - start
                    ramp_frac  = min(
                        pos / max(window_len * 0.3, 1),
                        (window_len - 1 - pos) / max(window_len * 0.3, 1),
                        1.0,
                    )
                    mults = _anomaly_multipliers(atype)
                    for kpi, mult in mults.items():
                        low, high = KPI_CONFIG[kpi]["ranges"][slice_type]
                        base   = new_cell[kpi]
                        target = base * mult
                        new_cell[kpi] = _clamp(
                            base + ramp_frac * (target - base),
                            low * 0.01, high * 60.0,
                        )
                    break

        cell_state = new_cell
        ts_str     = timestamps[t].strftime("%Y-%m-%d %H:%M:%S")

        # One row per UE
        for ue_idx, ue_id in enumerate(ue_ids):
            row = {
                "timestamp":  ts_str,
                "cell_id":    cell_id,
                "ue_id":      ue_id,
                "slice_type": slice_type,
                "latitude":   lat,
                "longitude":  lon,
            }
            for kpi, conf in KPI_CONFIG.items():
                low, high   = conf["ranges"][slice_type]
                ue_val      = cell_state[kpi] + ue_offsets[ue_idx][kpi]
                row[kpi]    = _round_kpi(_clamp(ue_val, low * 0.01, high * 60.0), conf)

            row["anomaly"]      = step_is_anomaly[t]
            row["anomaly_type"] = atype
            rows.append(row)

    return rows


# ============================================================================
# Main dataset generator
# ============================================================================

def generate_dashboard_dataset(
    num_cells,
    ues_per_cell,
    timestamps,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)

    T = len(timestamps)
    print(f"  Timesteps      : {T:,}  ({timestamps[0]}  →  {timestamps[-1]})")
    print(f"  Cells          : {num_cells}")
    print(f"  UEs per cell   : {ues_per_cell}")
    print(f"  Total rows     : {T * num_cells * ues_per_cell:,}")
    print()

    all_rows = []

    for c_idx in range(num_cells):
        cell_id    = f"gNB-{c_idx + 1:03d}"
        slice_type = random.choices(SLICE_TYPES, weights=SLICE_WEIGHTS, k=1)[0]
        lat        = round(random.uniform(*GEO_LAT_RANGE), 6)
        lon        = round(random.uniform(*GEO_LON_RANGE), 6)

        # Generate stable UE IDs for this cell
        ue_ids = [f"UE-{cell_id}-{uuid.uuid4().hex[:8].upper()}" for _ in range(ues_per_cell)]

        print(f"  [{c_idx+1:>3}/{num_cells}] {cell_id}  slice={slice_type:<5}  "
              f"lat={lat:.4f}  lon={lon:.4f}  UEs={ues_per_cell}")

        cell_rows = generate_cell_series(cell_id, slice_type, lat, lon, ue_ids, timestamps)
        all_rows.extend(cell_rows)

    return all_rows


# ============================================================================
# I/O helpers
# ============================================================================

def save_to_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved : {output_path}  ({size_mb:.1f} MB,  {len(rows):,} rows)")


def print_summary(rows):
    total      = len(rows)
    n_anomaly  = sum(r["anomaly"] for r in rows)
    n_normal   = total - n_anomaly
    cells      = sorted({r["cell_id"] for r in rows})
    ues        = {r["ue_id"] for r in rows}
    slices     = {}
    for r in rows:
        if r["cell_id"] not in slices:
            slices[r["cell_id"]] = r["slice_type"]

    print("\n" + "=" * 65)
    print("DASHBOARD DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total rows     : {total:,}")
    print(f"  Normal rows    : {n_normal:,}  ({n_normal/total*100:.1f}%)")
    print(f"  Anomaly rows   : {n_anomaly:,}  ({n_anomaly/total*100:.1f}%)")
    print(f"  Cells          : {len(cells)}")
    print(f"  Unique UEs     : {len(ues):,}")
    print(f"  Time range     : {rows[0]['timestamp']}  →  {rows[-1]['timestamp']}")
    print(f"\n  Slice distribution across cells :")
    slice_counts = {}
    for s in slices.values():
        slice_counts[s] = slice_counts.get(s, 0) + 1
    for s, cnt in sorted(slice_counts.items()):
        print(f"    {s:<6} : {cnt} cell(s)")
    print(f"\n  Anomaly type distribution :")
    ac = {}
    for r in rows:
        if r["anomaly"] == 1:
            ac[r["anomaly_type"]] = ac.get(r["anomaly_type"], 0) + 1
    for at, cnt in sorted(ac.items(), key=lambda x: -x[1]):
        print(f"    {at:<28} : {cnt:,}  ({cnt/n_anomaly*100:.1f}%)")
    print(f"\n  Columns ({len(rows[0])}) :")
    print(f"    IDs       : timestamp, cell_id, ue_id, slice_type, latitude, longitude")
    print(f"    KPIs ({len(KPI_COLUMNS)}) : {', '.join(KPI_COLUMNS)}")
    print(f"    Labels    : anomaly, anomaly_type")
    print("=" * 65)


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="5G Dashboard dataset generator — per-cell time series"
    )
    parser.add_argument("--num-cells",    type=int,   default=20,
                        help="Number of gNB cells (default: 20)")
    parser.add_argument("--ues-per-cell", type=int,   default=5,
                        help="UEs per cell (default: 5)")
    parser.add_argument("--start-date",   type=str,   default="2026-01-01 00:00:00",
                        help="Start datetime 'YYYY-MM-DD HH:MM:SS' (default: 2026-01-01 00:00:00)")
    parser.add_argument("--end-date",     type=str,   default="2026-04-01 00:00:00",
                        help="End datetime   'YYYY-MM-DD HH:MM:SS' (default: 2026-04-01 00:00:00)")
    parser.add_argument("--step-minutes", type=int,   default=5,
                        help="Interval between timesteps in minutes (default: 5)")
    parser.add_argument("--output",       type=str,   default="Data/Dashboard_data.csv",
                        help="Output CSV path (default: Data/Dashboard_data.csv)")
    parser.add_argument("--seed",         type=int,   default=42)

    args = parser.parse_args()

    fmt        = "%Y-%m-%d %H:%M:%S"
    start_dt   = datetime.strptime(args.start_date, fmt)
    end_dt     = datetime.strptime(args.end_date,   fmt)
    step       = timedelta(minutes=args.step_minutes)
    timestamps = []
    t          = start_dt
    while t < end_dt:
        timestamps.append(t)
        t += step

    total_rows = len(timestamps) * args.num_cells * args.ues_per_cell

    print("=" * 65)
    print("5G DASHBOARD DATASET GENERATOR — PER-CELL TIME SERIES")
    print("=" * 65)
    print(f"  Start          : {args.start_date}")
    print(f"  End            : {args.end_date}")
    print(f"  Step           : {args.step_minutes} min")
    print(f"  Cells          : {args.num_cells}")
    print(f"  UEs per cell   : {args.ues_per_cell}")
    print(f"  Timesteps      : {len(timestamps):,}")
    print(f"  Expected rows  : {total_rows:,}")
    print(f"  Output         : {args.output}")
    print()

    rows = generate_dashboard_dataset(
        num_cells=args.num_cells,
        ues_per_cell=args.ues_per_cell,
        timestamps=timestamps,
        seed=args.seed,
    )

    save_to_csv(rows, args.output)
    print_summary(rows)


if __name__ == "__main__":
    main()
