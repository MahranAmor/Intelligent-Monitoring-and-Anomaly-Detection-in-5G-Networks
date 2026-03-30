"""Kafka streaming configuration for the 5G anomaly monitoring pipeline."""

from pathlib import Path

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_MODEL_DATA = "model-data"
CONSUMER_GROUP = "5g-anomaly-consumer-group"

# Streaming defaults
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_ACK_TIMEOUT_SECONDS = 10
DEFAULT_MAX_MESSAGES = None


def default_model_data_csv_path() -> Path:
	"""Return the absolute path to Data/Model_data.csv from this package."""
	# config/ -> kafka_streaming/ -> project_root/
	return Path(__file__).resolve().parents[2] / "Data" / "Model_data.csv"
