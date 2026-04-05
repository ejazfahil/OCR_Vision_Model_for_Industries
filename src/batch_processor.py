"""Batch document processing with progress tracking."""
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
from src.pipeline import OCRPipeline


class BatchProcessor:
    """Process multiple documents and aggregate results."""

    def __init__(self, engine: str = "tesseract") -> None:
        self.pipeline = OCRPipeline(engine=engine)
        self.results: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def process_directory(self, directory: str) -> Dict[str, Any]:
        """Process all supported documents in a directory."""
        supported = {".pdf", ".jpg", ".jpeg", ".png", ".tiff"}
        files = [f for f in Path(directory).iterdir() if f.suffix.lower() in supported]
        self.results = []
        self.errors = []
        for f in files:
            try:
                result = self.pipeline.process(str(f))
                self.results.append(result)
            except Exception as e:
                self.errors.append(f"{f.name}: {e}")
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        return {
            "processed": len(self.results),
            "errors": len(self.errors),
            "avg_confidence": (
                sum(r["confidence"] for r in self.results) / len(self.results)
                if self.results else 0.0
            ),
            "error_list": self.errors,
        }
