"""Structured logging for OCR pipeline."""
import logging
import json
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "msg": %(message)s}'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_processing_event(logger: logging.Logger, file: str, duration_ms: float, confidence: float) -> None:
    logger.info(json.dumps({
        "event": "document_processed",
        "file": file,
        "duration_ms": round(duration_ms, 2),
        "confidence": round(confidence, 4),
        "timestamp": datetime.utcnow().isoformat()
    }))
