"""OCR engine abstraction layer."""
from __future__ import annotations
from typing import List, Dict, Any


class OCRResult:
    """Container for OCR extraction results."""

    def __init__(self, text: str, confidence: float, bbox: tuple) -> None:
        self.text = text
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }

    def __repr__(self) -> str:
        return f"OCRResult(text={self.text!r}, confidence={self.confidence:.2f})"


class MockOCREngine:
    """Mock OCR engine for testing without GPU/Tesseract dependency."""

    def __init__(self, engine: str = "tesseract") -> None:
        self.engine = engine

    def extract(self, image_path: str) -> List[OCRResult]:
        """Return mock extraction results."""
        return [
            OCRResult("Invoice No: 12345", 0.97, (10, 10, 200, 30)),
            OCRResult("Date: 2025-11-23", 0.95, (10, 40, 200, 60)),
            OCRResult("Total: €4,250.00", 0.91, (10, 70, 200, 90)),
        ]
