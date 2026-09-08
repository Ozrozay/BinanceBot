"""
Structured JSON-lines logger.

Observer only — touching this file must never affect trading decisions.
Every log record is written as a single JSON line to a rotating file.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Configure root logger:
      - JSON lines to logs/bot.jsonl (rotating, 10 MB × 5 files)
      - Human-readable to stdout
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # JSON file handler
    file_handler = RotatingFileHandler(
        f"{log_dir}/bot.jsonl", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    # Human-readable stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )
    root.addHandler(stream_handler)
