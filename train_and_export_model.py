"""
Script d'entraînement et d'export du modèle Gradient Boosting
pour la détection d'anomalies dans les réseaux 5G.

Ce script reproduit exactement le pipeline de preprocessing du notebook
`Models/modele_ML_pfa_GradientBoosting.ipynb` et exporte les modèles
entraînés au format .pkl pour une utilisation dans le pipeline Kafka.

Usage:
    python train_and_export_model.py
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# =====================================================================
# Configuration
# =====================================================================
DATA_PATH = os.path.join("Data", "Model_data.csv")
OUTPUT_DIR = os.path.join("kafka_streaming", "models")

# =====================================================================
# 1. Chargement des données
# =====================================================================
print("=" * 70)
print("CHARGEMENT DES DONNÉES")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✅ Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# =====================================================================
# 2. Preprocessing (identique au notebook)
# =====================================================================
print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

# 2.1 Extraction des composantes temporelles
df['year']   = df['timestamp'].dt.year
df['month']  = df['timestamp'].dt.month
df['day']    = df['timestamp'].dt.day
df['hour']   = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['second'] = df['timestamp'].dt.second
df['date']   = df['timestamp'].dt.date

# 2.2 Suppression des colonnes non-prédictives et corrélées
columns_to_drop = [
    'cell_id', 'ue_id', 'date',
    'packet_loss_rate_percent', 'rtt_ms',
    'energy_efficiency_bits_per_joule', 'reliability_percent',
    'throughput_ul_mbps', 'spectral_efficiency_bps_hz', 'bler_percent'
]
df.drop(columns=columns_to_drop, inplace=True)
print(f"Colonnes supprimées : {columns_to_drop}")

# 2.3 Encodage de slice_type (LabelEncoding)
le_slice = LabelEncoder()
df['slice_type'] = le_slice.fit_transform(df['slice_type'])
print(f"\nslice_type encodé :")
for label, encoded in zip(le_slice.classes_, le_slice.transform(le_slice.classes_)):
    print(f"   {label} -> {encoded}")

# 2.4 Copie propre
df_clean = df.copy()

# 2.5 Séparation Features / Targets
exclude_cols = ['timestamp', 'anomaly', 'anomaly_type']
X = df_clean.drop(columns=exclude_cols)

# Target binaire
y1 = df_clean['anomaly']

# Target multi-classes
le_anomaly = LabelEncoder()
y2 = le_anomaly.fit_transform(df_clean['anomaly_type'])

feature_columns = list(X.columns)
print(f"\nFeatures ({len(feature_columns)}) : {feature_columns}")
print(f"Target binaire (y1) : anomaly")
print(f"Target multi-classes (y2) : anomaly_type ({len(le_anomaly.classes_)} classes)")

# =====================================================================
# 3. Split Train/Test (80/20 stratifié)
# =====================================================================
print("\n" + "=" * 70)
print("SPLIT TRAIN/TEST")
print("=" * 70)

X_train_b, X_test_b, y1_train, y1_test = train_test_split(
    X, y1, test_size=0.2, random_state=42, stratify=y1
)
X_train_m, X_test_m, y2_train, y2_test = train_test_split(
    X, y2, test_size=0.2, random_state=42, stratify=y2
)

print(f"  Train : {X_train_b.shape[0]:,} samples")
print(f"  Test  : {X_test_b.shape[0]:,} samples")

# =====================================================================
# 4. Entraînement du modèle Gradient Boosting — Binaire
# =====================================================================
print("\n" + "=" * 70)
print("ENTRAÎNEMENT — Gradient Boosting BINAIRE")
print("=" * 70)

start = time.time()
gb_binary = GradientBoostingClassifier(random_state=42)
gb_binary.fit(X_train_b, y1_train)
elapsed_b = time.time() - start

y1_pred = gb_binary.predict(X_test_b)
y1_proba = gb_binary.predict_proba(X_test_b)[:, 1]

print(f"  ⏱️  Temps d'entraînement : {elapsed_b:.2f}s")
print(f"  Accuracy  : {accuracy_score(y1_test, y1_pred):.4f}")
print(f"  Precision : {precision_score(y1_test, y1_pred):.4f}")
print(f"  Recall    : {recall_score(y1_test, y1_pred):.4f}")
print(f"  F1-Score  : {f1_score(y1_test, y1_pred):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y1_test, y1_proba):.4f}")

# =====================================================================
# 5. Entraînement du modèle Gradient Boosting — Multi-classes
# =====================================================================
print("\n" + "=" * 70)
print("ENTRAÎNEMENT — Gradient Boosting MULTI-CLASSES")
print("=" * 70)

start = time.time()
gb_multi = GradientBoostingClassifier(random_state=42)
gb_multi.fit(X_train_m, y2_train)
elapsed_m = time.time() - start

y2_pred = gb_multi.predict(X_test_m)

print(f"  ⏱️  Temps d'entraînement : {elapsed_m:.2f}s")
print(f"  Accuracy  : {accuracy_score(y2_test, y2_pred):.4f}")
print(f"  Precision : {precision_score(y2_test, y2_pred, average='weighted'):.4f}")
print(f"  Recall    : {recall_score(y2_test, y2_pred, average='weighted'):.4f}")
print(f"  F1-Score  : {f1_score(y2_test, y2_pred, average='weighted'):.4f}")

# =====================================================================
# 6. Export des modèles en .pkl
# =====================================================================
print("\n" + "=" * 70)
print("EXPORT DES MODÈLES")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modèles
joblib.dump(gb_binary, os.path.join(OUTPUT_DIR, "gradient_boosting_binary.pkl"))
print(f"  ✅ gradient_boosting_binary.pkl")

joblib.dump(gb_multi, os.path.join(OUTPUT_DIR, "gradient_boosting_multiclass.pkl"))
print(f"  ✅ gradient_boosting_multiclass.pkl")

# Encodeurs
joblib.dump(le_slice, os.path.join(OUTPUT_DIR, "label_encoder_slice_type.pkl"))
print(f"  ✅ label_encoder_slice_type.pkl")

joblib.dump(le_anomaly, os.path.join(OUTPUT_DIR, "label_encoder_anomaly_type.pkl"))
print(f"  ✅ label_encoder_anomaly_type.pkl")

# Colonnes de features
joblib.dump(feature_columns, os.path.join(OUTPUT_DIR, "feature_columns.pkl"))
print(f"  ✅ feature_columns.pkl")

print(f"\n✅ Tous les fichiers exportés dans : {OUTPUT_DIR}/")
print("=" * 70)
