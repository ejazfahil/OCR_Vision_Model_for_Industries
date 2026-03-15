# API Reference

## OCRPipeline

```python
from src.pipeline import OCRPipeline

pipeline = OCRPipeline(engine="tesseract")  # or "easyocr"
result = pipeline.process("invoice.pdf")
```

### Returns
```json
{
  "file": "invoice.pdf",
  "engine": "tesseract",
  "confidence": 0.94,
  "fields": {
    "invoice_number": "INV-2025-001",
    "date": "2025-11-23",
    "amount": "€4,250.00"
  },
  "raw_text": "Invoice No: INV-2025-001 Date: 2025-11-23 Total: €4,250.00"
}
```

## FieldExtractor

```python
from src.field_extractor import extract_invoice_number, extract_date

text = "Invoice No: INV-001 Date: 2025-11-23"
invoice = extract_invoice_number(text)  # "INV-001"
date = extract_date(text)               # "2025-11-23"
```
