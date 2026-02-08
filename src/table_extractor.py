"""Table detection and extraction from document images."""
from __future__ import annotations
from typing import List, Dict, Any


class TableCell:
    def __init__(self, row: int, col: int, text: str) -> None:
        self.row = row
        self.col = col
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        return {"row": self.row, "col": self.col, "text": self.text}


class TableExtractor:
    """Extracts tabular data from OCR results."""

    def __init__(self, min_cols: int = 2) -> None:
        self.min_cols = min_cols

    def detect_rows(self, ocr_lines: List[str]) -> List[List[str]]:
        """Split lines into potential table rows based on spacing."""
        rows = []
        for line in ocr_lines:
            parts = [p.strip() for p in line.split("  ") if p.strip()]
            if len(parts) >= self.min_cols:
                rows.append(parts)
        return rows

    def to_csv_string(self, rows: List[List[str]]) -> str:
        """Convert rows to CSV string."""
        return "\n".join(",".join(row) for row in rows)
