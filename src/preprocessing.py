"""Image preprocessing utilities for OCR pipeline."""
from __future__ import annotations
import base64


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def validate_image_format(filename: str) -> bool:
    """Check if file extension is supported."""
    supported = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"}
    return any(filename.lower().endswith(ext) for ext in supported)


def normalize_confidence(raw_score: float) -> float:
    """Clamp confidence score to [0, 1]."""
    return max(0.0, min(1.0, raw_score))
