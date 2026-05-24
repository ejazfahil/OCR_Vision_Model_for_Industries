# OCR Benchmark Results

## Test Dataset
- 200 industrial invoices (printed)
- 150 utility meter readings (printed + handwritten)
- 100 maintenance manual pages (scanned)

## Results

### Character Error Rate (CER)
| Document Type | Tesseract | EasyOCR |
|--------------|-----------|----------|
| Printed invoice | 0.8% | 0.6% |
| Utility meter | 2.1% | 1.8% |
| Scanned manual | 3.4% | 2.9% |
| Handwritten | 8.5% | 6.2% |

### Processing Speed
| Engine | Avg per page | GPU required |
|--------|-------------|---------------|
| Tesseract | 120ms | No |
| EasyOCR | 180ms | Optional |

### Field Extraction Accuracy
| Field | Precision | Recall |
|-------|-----------|--------|
| Invoice number | 96.4% | 94.2% |
| Date | 98.1% | 97.8% |
| Amount | 95.7% | 93.5% |
