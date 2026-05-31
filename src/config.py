"""Pipeline configuration management."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class OCRConfig:
    """Configuration for the OCR pipeline."""
    engine: str = "tesseract"
    min_confidence: float = 0.7
    supported_formats: List[str] = field(
        default_factory=lambda: [".pdf", ".jpg", ".jpeg", ".png", ".tiff"]
    )
    dpi: int = 300
    enable_table_extraction: bool = True
    enable_field_extraction: bool = True
    batch_size: int = 10
    log_level: str = "INFO"

    def validate(self) -> None:
        if self.engine not in {"tesseract", "easyocr"}:
            raise ValueError(f"Unknown engine: {self.engine}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.dpi < 72:
            raise ValueError("DPI too low, minimum is 72")
