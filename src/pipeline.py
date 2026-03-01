"""End-to-end OCR pipeline orchestrator."""
from __future__ import annotations
from typing import Dict, Any, List
from src.preprocessing import validate_image_format, normalize_confidence
from src.ocr_engine import MockOCREngine, OCRResult
from src.field_extractor import extract_invoice_number, extract_date, extract_currency_amount


class OCRPipeline:
    """Orchestrates document → structured JSON extraction."""

    def __init__(self, engine: str = "tesseract") -> None:
        self.engine = MockOCREngine(engine=engine)

    def process(self, image_path: str) -> Dict[str, Any]:
        """Process a document image and return structured data."""
        if not validate_image_format(image_path):
            raise ValueError(f"Unsupported image format: {image_path}")

        results: List[OCRResult] = self.engine.extract(image_path)
        full_text = " ".join(r.text for r in results)
        avg_confidence = normalize_confidence(
            sum(r.confidence for r in results) / len(results) if results else 0
        )

        return {
            "file": image_path,
            "engine": self.engine.engine,
            "confidence": avg_confidence,
            "fields": {
                "invoice_number": extract_invoice_number(full_text),
                "date": extract_date(full_text),
                "amount": extract_currency_amount(full_text),
            },
            "raw_text": full_text,
        }
