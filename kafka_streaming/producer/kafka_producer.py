"""Kafka producer for 5G metrics streaming."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

KAFKA_STREAMING_DIR = Path(__file__).resolve().parents[1]
if str(KAFKA_STREAMING_DIR) not in sys.path:
	sys.path.insert(0, str(KAFKA_STREAMING_DIR))

from config.kafka_config import (
	BOOTSTRAP_SERVERS,
	DEFAULT_ACK_TIMEOUT_SECONDS,
	DEFAULT_DELAY_SECONDS,
	DEFAULT_MAX_MESSAGES,
	TOPIC_MODEL_DATA,
	default_model_data_csv_path,
)

REQUIRED_COLUMNS = {
	"timestamp",
	"cell_id",
	"ue_id",
	"slice_type",
	"anomaly",
	"anomaly_type",
}


class FiveGMetricsProducer:
	"""Kafka producer that streams 5G KPI records from a CSV file."""

	def __init__(
		self,
		bootstrap_servers: str = BOOTSTRAP_SERVERS,
		topic: str = TOPIC_MODEL_DATA,
		connect: bool = True,
	):
		self.bootstrap_servers = bootstrap_servers
		self.topic = topic
		self.producer: Optional[KafkaProducer] = None
		if connect:
			self._connect_producer()

	def _connect_producer(self) -> None:
		try:
			self.producer = KafkaProducer(
				bootstrap_servers=self.bootstrap_servers,
				acks="all",
				retries=5,
				linger_ms=5,
				value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
				key_serializer=lambda k: k.encode("utf-8") if k else None,
			)
		except NoBrokersAvailable as exc:
			raise RuntimeError(
				"No Kafka broker available. Start Kafka on "
				f"{self.bootstrap_servers} or run with --dry-run to validate CSV flow without Kafka."
			) from exc

	@staticmethod
	def _to_json_safe(value: Any) -> Any:
		if pd.isna(value):
			return None
		if isinstance(value, (datetime, pd.Timestamp)):
			return value.isoformat()
		if hasattr(value, "item") and not isinstance(value, (str, bytes)):
			try:
				return value.item()
			except Exception:
				return value
		return value

	@staticmethod
	def _normalize_key_part(part: Any) -> Optional[str]:
		if part is None:
			return None
		if isinstance(part, float) and pd.isna(part):
			return None
		text = str(part).strip()
		return text if text else None

	def _build_message_key(self, index: int, record: Dict[str, Any]) -> str:
		cell_key = self._normalize_key_part(record.get("cell_id"))
		ue_key = self._normalize_key_part(record.get("ue_id"))

		if cell_key and ue_key:
			return f"{cell_key}:{ue_key}:{index}"
		if cell_key:
			return f"{cell_key}:{index}"
		if ue_key:
			return f"{ue_key}:{index}"
		return str(index)

	@staticmethod
	def _validate_columns(df: pd.DataFrame) -> None:
		missing = sorted(REQUIRED_COLUMNS - set(df.columns))
		if missing:
			raise ValueError(
				"CSV is missing required columns: " + ", ".join(missing)
			)

	def stream_csv_data(
		self,
		csv_file: Optional[str] = None,
		delay: float = DEFAULT_DELAY_SECONDS,
		max_messages: Optional[int] = DEFAULT_MAX_MESSAGES,
		ack_timeout_seconds: int = DEFAULT_ACK_TIMEOUT_SECONDS,
		dry_run: bool = False,
	) -> Dict[str, int]:
		csv_path = Path(csv_file) if csv_file else default_model_data_csv_path()
		if not csv_path.exists():
			raise FileNotFoundError(f"CSV file not found: {csv_path}")

		df = pd.read_csv(csv_path, low_memory=False)
		if df.empty:
			raise ValueError(f"CSV file is empty: {csv_path}")

		self._validate_columns(df)

		if max_messages is not None:
			if max_messages <= 0:
				raise ValueError("max_messages must be > 0 when provided")
			df = df.head(max_messages)

		total = len(df)
		mode = "DRY-RUN" if dry_run else "KAFKA"
		print(f"Starting stream ({mode}): {total} records -> topic '{self.topic}'")
		print(f"Source CSV: {csv_path}")

		sent_count = 0
		failed_count = 0

		try:
			if not dry_run and self.producer is None:
				self._connect_producer()

			for index, row in df.iterrows():
				message = {col: self._to_json_safe(val) for col, val in row.to_dict().items()}
				message["stream_timestamp"] = datetime.now().isoformat()
				message["record_id"] = int(index)

				key = self._build_message_key(index=index, record=message)

				if dry_run:
					sent_count += 1
					print(f"DRY-RUN message {index} prepared with key '{key}'")
					if delay > 0:
						time.sleep(delay)
					continue

				future = self.producer.send(self.topic, key=key, value=message)  # type: ignore[union-attr]

				try:
					metadata = future.get(timeout=ack_timeout_seconds)
					sent_count += 1
					print(
						f"OK message {index} - partition: {metadata.partition}, offset: {metadata.offset}"
					)
				except Exception as exc:
					failed_count += 1
					print(f"ERROR message {index}: {exc}")

				if delay > 0:
					time.sleep(delay)
		except KeyboardInterrupt:
			print("\nStreaming interrupted by user.")
		finally:
			if self.producer is not None:
				self.producer.flush()
				self.producer.close()

		stats = {
			"sent": sent_count,
			"failed": failed_count,
			"total": total,
		}
		print(f"\nStreaming done: {stats}")
		return stats


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Stream 5G metrics from CSV to Kafka")
	parser.add_argument("--bootstrap-servers", default=BOOTSTRAP_SERVERS)
	parser.add_argument("--topic", default=TOPIC_MODEL_DATA)
	parser.add_argument("--csv-file", default=str(default_model_data_csv_path()))
	parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
	parser.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
	parser.add_argument("--ack-timeout", type=int, default=DEFAULT_ACK_TIMEOUT_SECONDS)
	parser.add_argument("--dry-run", action="store_true", help="Validate CSV and message formatting without Kafka")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	producer = FiveGMetricsProducer(
		bootstrap_servers=args.bootstrap_servers,
		topic=args.topic,
		connect=not args.dry_run,
	)
	producer.stream_csv_data(
		csv_file=args.csv_file,
		delay=args.delay,
		max_messages=args.max_messages,
		ack_timeout_seconds=args.ack_timeout,
		dry_run=args.dry_run,
	)
