"""
Générateur de dataset synthétique pour la détection d'anomalies des réseaux 5G.
Projet PFA - Ayoub & Mahran

Ce script génère un jeu de données réaliste contenant des mesures KPI 5G
avec des échantillons normaux et anomaliques, prêt pour l'entraînement
de modèles de détection d'anomalies.

Usage:
    python3 generate_5g_dataset.py
    python3 generate_5g_dataset.py --num-records 200000 --anomaly-ratio 0.08
"""

import argparse
import csv
import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np

# ============================================================================
# Configuration des KPIs par type de slice (eMBB, URLLC, mMTC)
# Chaque KPI définit : (min_normal, max_normal) pour la plage normale
# ============================================================================

KPI_CONFIG = {
    # --- A. Latency & Delay Performance ---
    "one_way_latency_ms": {
        "kpi_id": "KPI-1",
        "unit": "ms",
        "description": "One-way latency",
        "ranges": {
            "eMBB":  (1.0, 20.0),
            "URLLC": (0.1, 5.0),
            "mMTC":  (10.0, 100.0),
        },
    },
    "jitter_ms": {
        "kpi_id": "KPI-2",
        "unit": "ms",
        "description": "Variation of one-way latency",
        "ranges": {
            "eMBB":  (1.0, 10.0),
            "URLLC": (0.01, 1.0),
            "mMTC":  (5.0, 20.0),
        },
    },
    "rtt_ms": {
        "kpi_id": "KPI-3",
        "unit": "ms",
        "description": "Round Trip Time",
        "ranges": {
            "eMBB":  (10.0, 40.0),
            "URLLC": (0.5, 10.0),
            "mMTC":  (20.0, 200.0),
        },
    },
    "packet_delay_budget_ms": {
        "kpi_id": "KPI-18",
        "unit": "ms",
        "description": "Packet Delay Budget",
        "ranges": {
            "eMBB":  (50.0, 100.0),
            "URLLC": (0.5, 1.0),
            "mMTC":  (50.0, 100.0),
        },
    },
    "handover_interruption_time_ms": {
        "kpi_id": "KPI-16",
        "unit": "ms",
        "description": "Handover Interruption Time",
        "ranges": {
            "eMBB":  (5.0, 50.0),
            "URLLC": (0.5, 10.0),
            "mMTC":  (10.0, 60.0),
        },
    },
    # --- B. Reliability & Transmission Quality ---
    "reliability_percent": {
        "kpi_id": "KPI-4",
        "unit": "%",
        "description": "Reliability (packet delivery ratio)",
        "ranges": {
            "eMBB":  (99.0, 100.0),
            "URLLC": (99.999, 100.0),
            "mMTC":  (95.0, 100.0),
        },
    },
    "packet_loss_percent": {
        "kpi_id": "KPI-5",
        "unit": "%",
        "description": "Packet Loss",
        "ranges": {
            "eMBB":  (0.0, 1.0),
            "URLLC": (0.0, 0.001),
            "mMTC":  (0.0, 5.0),
        },
    },
    "packet_loss_rate_percent": {
        "kpi_id": "KPI-10",
        "unit": "%",
        "description": "Packet Loss Rate",
        "ranges": {
            "eMBB":  (0.0, 1.0),
            "URLLC": (0.0, 0.001),
            "mMTC":  (0.0, 5.0),
        },
    },
    "bler_percent": {
        "kpi_id": "KPI-14",
        "unit": "%",
        "description": "Block Error Rate",
        "ranges": {
            "eMBB":  (0.0, 10.0),
            "URLLC": (0.0, 1.0),
            "mMTC":  (0.0, 15.0),
        },
    },
    # --- C. Capacity & Throughput ---
    "throughput_dl_mbps": {
        "kpi_id": "KPI-6",
        "unit": "Mbps",
        "description": "Throughput Downlink",
        "ranges": {
            "eMBB":  (100.0, 20000.0),
            "URLLC": (10.0, 200.0),
            "mMTC":  (0.01, 1.0),
        },
    },
    "throughput_ul_mbps": {
        "kpi_id": "KPI-7",
        "unit": "Mbps",
        "description": "Throughput Uplink / Area",
        "ranges": {
            "eMBB":  (50.0, 10000.0),
            "URLLC": (10.0, 200.0),
            "mMTC":  (0.01, 1.0),
        },
    },
    "spectral_efficiency_bps_hz": {
        "kpi_id": "KPI-9",
        "unit": "bits/s/Hz",
        "description": "Spectral Efficiency",
        "ranges": {
            "eMBB":  (10.0, 30.0),
            "URLLC": (5.0, 15.0),
            "mMTC":  (1.0, 5.0),
        },
    },
    # --- D. Mobility & Network Continuity ---
    "handover_success_rate_percent": {
        "kpi_id": "KPI-15",
        "unit": "%",
        "description": "Handover Success Rate",
        "ranges": {
            "eMBB":  (99.0, 100.0),
            "URLLC": (99.0, 100.0),
            "mMTC":  (99.0, 100.0),
        },
    },
    # --- E. Energy & Efficiency ---
    "energy_efficiency_bits_per_joule": {
        "kpi_id": "KPI-17",
        "unit": "bits/J",
        "description": "Energy Efficiency",
        "ranges": {
            "eMBB":  (1e6, 1e8),
            "URLLC": (1e7, 1e9),
            "mMTC":  (1e4, 1e6),
        },
    },
}

# Liste des noms de colonnes KPI (dans l'ordre)
KPI_COLUMNS = list(KPI_CONFIG.keys())

# ============================================================================
# Types d'anomalies réseau 5G
# ============================================================================

ANOMALY_TYPES = [
    "network_congestion",       # Congestion réseau
    "interference",             # Interférence radio
    "hardware_failure",         # Défaillance matérielle
    "handover_failure",         # Échec de handover
    "signal_degradation",       # Dégradation du signal
    "security_attack",          # Attaque de sécurité (DDoS, jamming)
    "backhaul_issue",           # Problème de backhaul
    "overload",                 # Surcharge de la cellule
]

# ============================================================================
# Configuration du réseau simulé
# ============================================================================

NUM_CELLS = 50           # Nombre de stations de base (gNB)
NUM_UES = 500            # Nombre d'équipements utilisateurs
SLICE_TYPES = ["eMBB", "URLLC", "mMTC"]
SLICE_WEIGHTS = [0.50, 0.30, 0.20]  # Distribution des types de slice

# Zone géographique simulée (région rectangulaire ~ ville)
GEO_LAT_RANGE = (33.5, 34.1)    # Latitude (ex: Casablanca)
GEO_LON_RANGE = (-7.7, -7.4)    # Longitude


def generate_cell_topology(num_cells):
    """Génère la topologie des cellules (stations de base gNB)."""
    cells = []
    for i in range(num_cells):
        cell_id = f"gNB-{i+1:03d}"
        lat = random.uniform(*GEO_LAT_RANGE)
        lon = random.uniform(*GEO_LON_RANGE)
        cells.append({"cell_id": cell_id, "lat": lat, "lon": lon})
    return cells


def generate_ue_pool(num_ues):
    """Génère un pool d'identifiants UE anonymisés."""
    return [f"UE-{uuid.uuid4().hex[:8].upper()}" for _ in range(num_ues)]


def generate_normal_kpi_values(slice_type):
    """
    Génère un enregistrement KPI normal pour un type de slice donné.
    Les valeurs suivent des distributions réalistes à l'intérieur des plages normales.
    """
    record = {}
    for kpi_name, kpi_conf in KPI_CONFIG.items():
        low, high = kpi_conf["ranges"][slice_type]

        # Utiliser une distribution qui concentre les valeurs vers le centre
        # avec une queue légère vers les bords (distribution beta)
        if high > low:
            # Distribution beta centrée (alpha=2, beta=2 -> forme en cloche)
            beta_sample = np.random.beta(2.5, 2.5)
            value = low + beta_sample * (high - low)
        else:
            value = low

        # Arrondir selon l'unité
        if kpi_conf["unit"] == "%" and high <= 1.0:
            value = round(value, 6)
        elif kpi_conf["unit"] == "bits/J":
            value = round(value, 0)
        else:
            value = round(value, 4)

        record[kpi_name] = value
    return record


def apply_anomaly(record, slice_type, anomaly_type):
    """
    Applique une anomalie réaliste à un enregistrement KPI.
    Chaque type d'anomalie affecte un sous-ensemble spécifique de KPIs
    avec des multiplicateurs réalistes.
    """
    r = record.copy()

    if anomaly_type == "network_congestion":
        # Latence élevée, throughput réduit, perte de paquets accrue
        factor = random.uniform(2.0, 5.0)
        r["one_way_latency_ms"] *= factor
        r["rtt_ms"] *= factor
        r["jitter_ms"] *= random.uniform(2.0, 4.0)
        r["packet_delay_budget_ms"] *= random.uniform(1.5, 3.0)
        r["throughput_dl_mbps"] *= random.uniform(0.1, 0.4)
        r["throughput_ul_mbps"] *= random.uniform(0.1, 0.4)
        r["packet_loss_percent"] = min(r["packet_loss_percent"] * random.uniform(5, 20), 50.0)
        r["packet_loss_rate_percent"] = min(r["packet_loss_rate_percent"] * random.uniform(5, 20), 50.0)

    elif anomaly_type == "interference":
        # BLER élevé, efficacité spectrale réduite, throughput impacté
        r["bler_percent"] = min(r["bler_percent"] * random.uniform(3, 8), 80.0)
        r["spectral_efficiency_bps_hz"] *= random.uniform(0.2, 0.5)
        r["throughput_dl_mbps"] *= random.uniform(0.2, 0.5)
        r["throughput_ul_mbps"] *= random.uniform(0.3, 0.6)
        r["jitter_ms"] *= random.uniform(1.5, 3.0)
        r["packet_loss_percent"] = min(r["packet_loss_percent"] + random.uniform(2, 10), 40.0)
        r["packet_loss_rate_percent"] = min(r["packet_loss_rate_percent"] + random.uniform(2, 10), 40.0)

    elif anomaly_type == "hardware_failure":
        # Dégradation sévère de tous les métriques
        r["reliability_percent"] = max(r["reliability_percent"] * random.uniform(0.85, 0.95), 50.0)
        r["throughput_dl_mbps"] *= random.uniform(0.05, 0.2)
        r["throughput_ul_mbps"] *= random.uniform(0.05, 0.2)
        r["one_way_latency_ms"] *= random.uniform(5, 15)
        r["rtt_ms"] *= random.uniform(5, 15)
        r["packet_loss_percent"] = min(r["packet_loss_percent"] + random.uniform(10, 30), 60.0)
        r["packet_loss_rate_percent"] = min(r["packet_loss_rate_percent"] + random.uniform(10, 30), 60.0)
        r["bler_percent"] = min(r["bler_percent"] + random.uniform(20, 50), 90.0)
        r["energy_efficiency_bits_per_joule"] *= random.uniform(0.1, 0.3)

    elif anomaly_type == "handover_failure":
        # Échec de handover, interruption prolongée
        r["handover_success_rate_percent"] = max(
            r["handover_success_rate_percent"] - random.uniform(10, 40), 40.0
        )
        r["handover_interruption_time_ms"] *= random.uniform(5, 20)
        r["one_way_latency_ms"] *= random.uniform(2, 6)
        r["rtt_ms"] *= random.uniform(2, 6)
        r["packet_loss_percent"] = min(r["packet_loss_percent"] + random.uniform(3, 15), 40.0)
        r["packet_loss_rate_percent"] = min(r["packet_loss_rate_percent"] + random.uniform(3, 15), 40.0)

    elif anomaly_type == "signal_degradation":
        # Dégradation progressive du signal
        r["spectral_efficiency_bps_hz"] *= random.uniform(0.3, 0.6)
        r["throughput_dl_mbps"] *= random.uniform(0.3, 0.6)
        r["throughput_ul_mbps"] *= random.uniform(0.3, 0.6)
        r["bler_percent"] = min(r["bler_percent"] * random.uniform(2, 5), 60.0)
        r["reliability_percent"] = max(r["reliability_percent"] - random.uniform(2, 8), 70.0)
        r["jitter_ms"] *= random.uniform(1.5, 3.0)

    elif anomaly_type == "security_attack":
        # DDoS / Jamming : latence extrême, perte massive
        r["one_way_latency_ms"] *= random.uniform(10, 50)
        r["rtt_ms"] *= random.uniform(10, 50)
        r["jitter_ms"] *= random.uniform(5, 20)
        r["packet_loss_percent"] = min(r["packet_loss_percent"] + random.uniform(20, 50), 80.0)
        r["packet_loss_rate_percent"] = min(r["packet_loss_rate_percent"] + random.uniform(20, 50), 80.0)
        r["throughput_dl_mbps"] *= random.uniform(0.01, 0.1)
        r["throughput_ul_mbps"] *= random.uniform(0.01, 0.1)
        r["reliability_percent"] = max(r["reliability_percent"] - random.uniform(10, 30), 30.0)

    elif anomaly_type == "backhaul_issue":
        # Problème de liaison backhaul : latence élevée, throughput réduit
        r["one_way_latency_ms"] *= random.uniform(3, 10)
        r["rtt_ms"] *= random.uniform(3, 10)
        r["packet_delay_budget_ms"] *= random.uniform(2, 5)
        r["throughput_dl_mbps"] *= random.uniform(0.1, 0.3)
        r["throughput_ul_mbps"] *= random.uniform(0.1, 0.3)
        r["jitter_ms"] *= random.uniform(3, 8)

    elif anomaly_type == "overload":
        # Surcharge cellule : dégradation modérée mais étendue
        r["throughput_dl_mbps"] *= random.uniform(0.2, 0.5)
        r["throughput_ul_mbps"] *= random.uniform(0.2, 0.5)
        r["one_way_latency_ms"] *= random.uniform(2, 4)
        r["rtt_ms"] *= random.uniform(2, 4)
        r["packet_loss_percent"] = min(r["packet_loss_percent"] + random.uniform(1, 8), 30.0)
        r["packet_loss_rate_percent"] = min(r["packet_loss_rate_percent"] + random.uniform(1, 8), 30.0)
        r["bler_percent"] = min(r["bler_percent"] * random.uniform(1.5, 3), 40.0)
        r["reliability_percent"] = max(r["reliability_percent"] - random.uniform(1, 5), 80.0)

    # Arrondir toutes les valeurs
    for kpi_name, kpi_conf in KPI_CONFIG.items():
        if kpi_conf["unit"] == "bits/J":
            r[kpi_name] = round(r[kpi_name], 0)
        elif kpi_conf["unit"] == "%" and kpi_conf["ranges"][slice_type][1] <= 1.0:
            r[kpi_name] = round(r[kpi_name], 6)
        else:
            r[kpi_name] = round(r[kpi_name], 4)

    return r


def generate_dataset(num_records, anomaly_ratio, seed=42):
    """
    Génère le dataset complet.

    Args:
        num_records: Nombre total d'enregistrements à générer
        anomaly_ratio: Proportion d'anomalies (ex: 0.05 = 5%)
        seed: Graine pour la reproductibilité

    Returns:
        Liste de dictionnaires représentant les enregistrements
    """
    random.seed(seed)
    np.random.seed(seed)

    cells = generate_cell_topology(NUM_CELLS)
    ue_pool = generate_ue_pool(NUM_UES)

    # Période de collecte : 30 jours avec un pas de 5 minutes
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    time_step = timedelta(minutes=5)

    num_anomalies = int(num_records * anomaly_ratio)
    num_normal = num_records - num_anomalies

    # Indices des enregistrements anomaliques (aléatoires)
    anomaly_indices = set(random.sample(range(num_records), num_anomalies))

    dataset = []

    for i in range(num_records):
        # Timestamp avec léger bruit
        timestamp = start_time + time_step * i + timedelta(seconds=random.randint(-30, 30))

        # Sélection aléatoire de la cellule, UE et slice
        cell = random.choice(cells)
        ue_id = random.choice(ue_pool)
        slice_type = random.choices(SLICE_TYPES, weights=SLICE_WEIGHTS, k=1)[0]

        # Coordonnées UE : proches de la cellule avec bruit gaussien
        ue_lat = cell["lat"] + np.random.normal(0, 0.005)
        ue_lon = cell["lon"] + np.random.normal(0, 0.005)

        # Générer les valeurs KPI normales
        kpi_values = generate_normal_kpi_values(slice_type)

        # Déterminer si cet enregistrement est anomalique
        is_anomaly = i in anomaly_indices
        anomaly_type = "normal"

        if is_anomaly:
            anomaly_type = random.choice(ANOMALY_TYPES)
            kpi_values = apply_anomaly(kpi_values, slice_type, anomaly_type)

        # Construire l'enregistrement
        record = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "cell_id": cell["cell_id"],
            "ue_id": ue_id,
            "slice_type": slice_type,
            "latitude": round(ue_lat, 6),
            "longitude": round(ue_lon, 6),
        }
        record.update(kpi_values)
        record["anomaly"] = 1 if is_anomaly else 0
        record["anomaly_type"] = anomaly_type

        dataset.append(record)

        if (i + 1) % 50000 == 0:
            print(f"  {i + 1:,} / {num_records:,} enregistrements générés...")

    return dataset


def save_to_csv(dataset, output_path):
    """Sauvegarde le dataset au format CSV."""
    if not dataset:
        print("Aucun enregistrement à sauvegarder.")
        return

    fieldnames = list(dataset[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Dataset sauvegardé : {output_path} ({file_size_mb:.1f} Mo)")


def print_summary(dataset):
    """Affiche un résumé du dataset généré."""
    total = len(dataset)
    anomalies = sum(1 for r in dataset if r["anomaly"] == 1)
    normal = total - anomalies

    print("\n" + "=" * 60)
    print("RÉSUMÉ DU DATASET GÉNÉRÉ")
    print("=" * 60)
    print(f"  Total d'enregistrements : {total:,}")
    print(f"  Enregistrements normaux : {normal:,} ({normal/total*100:.1f}%)")
    print(f"  Anomalies               : {anomalies:,} ({anomalies/total*100:.1f}%)")

    # Distribution des types de slice
    print(f"\n  Distribution des slices :")
    for st in SLICE_TYPES:
        count = sum(1 for r in dataset if r["slice_type"] == st)
        print(f"    {st:6s} : {count:,} ({count/total*100:.1f}%)")

    # Distribution des types d'anomalies
    print(f"\n  Distribution des anomalies :")
    anomaly_counts = {}
    for r in dataset:
        if r["anomaly"] == 1:
            at = r["anomaly_type"]
            anomaly_counts[at] = anomaly_counts.get(at, 0) + 1
    for at, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
        print(f"    {at:25s} : {count:,} ({count/anomalies*100:.1f}%)")

    # Aperçu des colonnes
    print(f"\n  Colonnes ({len(dataset[0])} total) :")
    print(f"    Dimensions : timestamp, cell_id, ue_id, slice_type, latitude, longitude")
    print(f"    KPIs ({len(KPI_COLUMNS)}) : {', '.join(KPI_COLUMNS)}")
    print(f"    Labels     : anomaly, anomaly_type")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Générateur de dataset synthétique 5G pour la détection d'anomalies"
    )
    parser.add_argument(
        "--num-records", type=int, default=300000,
        help="Nombre total d'enregistrements (défaut: 100000)"
    )
    parser.add_argument(
        "--anomaly-ratio", type=float, default=0.05,
        help="Proportion d'anomalies, entre 0 et 1 (défaut: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--output", type=str, default="5g_anomaly_dataset.csv",
        help="Chemin du fichier CSV de sortie (défaut: 5g_anomaly_dataset.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine aléatoire pour la reproductibilité (défaut: 42)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GÉNÉRATION DU DATASET 5G - DÉTECTION D'ANOMALIES")
    print("=" * 60)
    print(f"  Enregistrements : {args.num_records:,}")
    print(f"  Ratio anomalies : {args.anomaly_ratio*100:.1f}%")
    print(f"  Fichier sortie  : {args.output}")
    print(f"  Seed            : {args.seed}")
    print()

    dataset = generate_dataset(args.num_records, args.anomaly_ratio, args.seed)
    save_to_csv(dataset, args.output)
    print_summary(dataset)


if __name__ == "__main__":
    main()
