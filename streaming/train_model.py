import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLUMNS = [
    "one_way_latency_ms",
    "jitter_ms",
    "rtt_ms",
    "packet_delay_budget_ms",
    "handover_interruption_time_ms",
    "reliability_percent",
    "packet_loss_percent",
    "packet_loss_rate_percent",
    "bler_percent",
    "throughput_dl_mbps",
    "throughput_ul_mbps",
    "spectral_efficiency_bps_hz",
    "handover_success_rate_percent",
    "energy_efficiency_bits_per_joule",
    "slice_type_encoded",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "test_data.csv")
MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "models", "decision_tree_model.pkl")


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows.")

    # Encode slice_type
    le = LabelEncoder()
    df["slice_type_encoded"] = le.fit_transform(df["slice_type"])
    print(f"Slice type classes: {list(le.classes_)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["anomaly"]

    # Fill NaN with median
    X = X.fillna(X.median())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train Decision Tree
    clf = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    print("Model trained.")

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS)
    print("\nTop 5 Feature Importances:")
    print(importances.nlargest(5).to_string())

    # Save model bundle
    bundle = {
        "model": clf,
        "feature_columns": FEATURE_COLUMNS,
        "label_encoder": le,
        "label_encoder_classes": list(le.classes_),
    }
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nModel saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
