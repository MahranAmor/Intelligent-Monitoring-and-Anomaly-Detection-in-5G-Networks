# Kafka Streaming — Détection d'Anomalies 5G

Pipeline de streaming en temps réel utilisant Apache Kafka et un modèle Gradient Boosting pour la détection d'anomalies dans les réseaux 5G.

## Structure du projet

```
kafka_streaming/
├── config/
│   └── kafka_config.py          # Configuration Kafka (broker, topics, consumer group)
├── producer/
│   └── kafka_producer.py        # Producteur : envoie les métriques 5G vers Kafka
├── consumer/
│   └── kafka_consumer.py        # Consommateur : reçoit les messages et exécute l'inférence ML
├── ml/
│   └── model_loader.py          # Charge les modèles .pkl et fournit l'interface de prédiction
├── utils/
│   └── preprocessing.py         # Fonctions de preprocessing (identiques au pipeline d'entraînement)
├── models/                      # Fichiers modèles .pkl exportés
├── requirements.txt             # Dépendances Python
└── README.md                    # Ce fichier
```

## Prérequis

1. Apache Kafka installé et en cours d'exécution
2. Python 3.8+
3. Modèles exportés (exécuter `train_and_export_model.py` depuis la racine du projet)

## Installation

```bash
pip install -r requirements.txt
```

## Modèles disponibles

- `gradient_boosting_binary.pkl` — Classification binaire (Normal vs Anomaly)
- `gradient_boosting_multiclass.pkl` — Classification multi-classes (type d'anomalie)
- `label_encoder_slice_type.pkl` — Encodeur pour slice_type
- `label_encoder_anomaly_type.pkl` — Encodeur pour anomaly_type
- `feature_columns.pkl` — Liste des colonnes de features utilisées à l'entraînement
