# OCR Vision Model — Architecture

## Pipeline Overview

```
Input (PDF/Image)
    │
    ▼
Preprocessing
  ├── Format validation
  ├── Resolution normalization (300 DPI)
  └── Contrast enhancement
    │
    ▼
Layout Detection (LayoutLMv3)
  ├── Header / footer regions
  ├── Table detection
  └── Text block segmentation
    │
    ▼
OCR Engine (Tesseract / EasyOCR)
  ├── Line-level text extraction
  └── Confidence scoring per word
    │
    ▼
Post-processing
  ├── Named entity extraction
  ├── Field normalization
  └── JSON output
```

## Key Design Decisions

1. **Tesseract** for standard documents (fast, free)
2. **EasyOCR** fallback for handwriting and degraded scans
3. **LayoutLMv3** for structured document understanding
