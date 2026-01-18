"""Named entity and field extraction from OCR output."""
from __future__ import annotations
import re
from typing import Optional


def extract_invoice_number(text: str) -> Optional[str]:
    """Extract invoice number from OCR text."""
    match = re.search(r'(?i)invoice\s*(?:no|number|#)?[:\s]+(\w+)', text)
    return match.group(1) if match else None


def extract_date(text: str) -> Optional[str]:
    """Extract date in common formats."""
    patterns = [
        r'\d{4}-\d{2}-\d{2}',        # ISO
        r'\d{2}/\d{2}/\d{4}',        # US
        r'\d{2}\.\d{2}\.\d{4}',     # EU
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def extract_currency_amount(text: str) -> Optional[str]:
    """Extract monetary amounts with currency symbols."""
    match = re.search(r'(?:[€$£¥]|EUR|USD|GBP)\s*[\d,]+(?:\.\d{2})?', text)
    return match.group(0) if match else None
