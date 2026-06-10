# 🏭 OCR Vision Model for Industries

> A modular OCR pipeline for industrial document intelligence — extracting structured fields from invoices, datasheets, and **utility meter readings** using a multi-engine OCR ensemble, optional LLM verification, and a clean evaluation toolkit.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

**Status:** Architecture-complete framework with a real evaluation dataset. The pipeline, field/table extraction, CER/WER metrics, batch processing, and the multi-engine ensemble + LLM-verifier integrations are all implemented. By default the pipeline runs against a **mock OCR engine** (so the test suite and demo run with zero heavy dependencies); plugging in the real PaddleOCR / TrOCR / EasyOCR backends requires installing those optional packages. The accuracy figures in `docs/` are **illustrative design targets** (with AI-rendered charts), not yet reproduced by a committed end-to-end benchmark — so this README does not present them as measured results.

---

## Overview

Industrial documents — invoices, maintenance manuals, datasheets, and analog **utility meters** — encode high-value data in formats that defeat naïve OCR. This project provides an extensible pipeline that:

1. **Preprocesses** images (format validation, enhancement).
2. **Recognizes** text via a configurable OCR backend (mock by default; PaddleOCR / TrOCR / EasyOCR ensemble in production).
3. **Verifies** results with rule-based validation and (optionally) a multimodal LLM.
4. **Extracts** structured fields (invoice numbers, dates, amounts) and **tables**.
5. **Scores** output against ground truth with CER / WER.

This repository ships with a real-world target task: **927 cropped water-meter images** (`meter_images_jpg/`) plus a ground-truth spreadsheet (`water_meter_reading.xlsx`).

## Architecture

```
                ┌─────────────────────────────────────────────┐
   image ─────► │ preprocessing (validate + enhance)          │
                └───────────────────────┬─────────────────────┘
                                        ▼
                ┌─────────────────────────────────────────────┐
                │ OCR engine                                   │
                │  • MockOCREngine (default, dependency-free)  │
                │  • EnsembleOCR: PaddleOCR-VL · TrOCR · EasyOCR│
                │    with weighted / majority / highest voting │
                └───────────────────────┬─────────────────────┘
                                        ▼
                ┌─────────────────────────────────────────────┐
                │ LLMVerifier (optional)                       │
                │  • rule-based validation (length/range/regex)│
                │  • GPT-4V correction of confusable chars     │
                └───────────────────────┬─────────────────────┘
                                        ▼
        field_extractor (regex) ──► table_extractor ──► metrics (CER/WER) ──► export
```

## Key Modules

| Module | What it does | Real / runnable today |
|--------|--------------|-----------------------|
| `src/pipeline.py` | Orchestrates image → structured JSON | ✅ (mock engine by default) |
| `src/ocr_engines/ensemble_ocr.py` | PaddleOCR / TrOCR / EasyOCR with confidence-weighted, majority, and highest-confidence voting | ✅ code complete; needs optional OCR deps installed |
| `src/ocr_engines/llm_verifier.py` | Rule-based validation + GPT-4V error correction (e.g. `O`→`0`, `I`→`1`) | ✅ code complete; rules run offline, LLM needs an API key |
| `src/metrics.py` | Levenshtein-based **CER** and **WER** | ✅ fully functional |
| `src/field_extractor.py` | Regex extraction of invoice number / date / currency amount | ✅ fully functional |
| `src/table_extractor.py`, `src/batch_processor.py`, `src/export.py`, `src/preprocessing/` | Table parsing, directory batch processing, result export, image enhancement | ✅ implemented |

## Tech Stack & Tools

| Area | Tools |
|------|-------|
| Core (installed) | **OpenCV**, **Pillow**, **pandas**, **NumPy** |
| OCR backends (optional) | **PaddleOCR-VL**, **TrOCR** (`transformers`, `microsoft/trocr-base-printed`), **EasyOCR** |
| LLM verification (optional) | **OpenAI** GPT-4V / GPT-4 |
| Metrics | Custom CER / WER (edit distance) |
| Tooling | **pytest** + **pytest-cov**, **Makefile**, GitHub Actions CI |

## Dataset

- **Water meter images:** `meter_images_jpg/` — **927** cropped meter photographs (`img_*.jpg`).
- **Ground truth:** `water_meter_reading.xlsx` — reference readings for evaluation.

This makes the repo set up for a concrete CER/WER benchmark of meter-digit recognition once a real OCR backend is enabled.

## Project Structure

```
OCR_Vision_Model_for_Industries/
├── src/
│   ├── pipeline.py              # orchestration (mock engine default)
│   ├── ocr_engine.py            # OCRResult + MockOCREngine
│   ├── ocr_engines/
│   │   ├── ensemble_ocr.py      # PaddleOCR + TrOCR + EasyOCR voting
│   │   └── llm_verifier.py      # rule + GPT-4V verification
│   ├── preprocessing/           # image enhancement
│   ├── field_extractor.py       # regex field extraction
│   ├── table_extractor.py       # table parsing
│   ├── metrics.py               # CER / WER
│   ├── batch_processor.py       # directory-level processing
│   └── export.py
├── tests/                       # pytest suite across modules
├── docs/                        # API, ARCHITECTURE, BENCHMARKS (illustrative)
├── meter_images_jpg/            # 927 meter images
├── water_meter_reading.xlsx     # ground-truth readings
├── OCR_Interactive_Testing.ipynb
├── requirements.txt
├── LICENSE                      # MIT
└── README.md
```

## Getting Started

```bash
git clone https://github.com/ejazfahil/OCR_Vision_Model_for_Industries.git
cd OCR_Vision_Model_for_Industries
pip install -r requirements.txt        # core deps; runs with the mock engine

# run the test suite
pytest tests/
```

```python
from src.pipeline import OCRPipeline

pipeline = OCRPipeline()               # mock engine by default
result = pipeline.process("invoice.png")
print(result["fields"])                # {'invoice_number': ..., 'date': ..., 'amount': ...}
```

To run with real OCR, install the optional backends (`paddleocr`, `easyocr`, `transformers`) and an LLM key, then route the pipeline through `EnsembleOCR` / `LLMVerifier`.

## Key Features

- **Pluggable engine layer** — swap mock ↔ ensemble without touching downstream code.
- **Confidence-weighted ensemble voting** across three SOTA OCR models.
- **LLM-assisted correction** targeting the confusable-character errors typical of meter OCR.
- **Domain-aware validation** — expected digit length, numeric-only, value range, regex pattern.
- **Honest metrics** — real edit-distance CER/WER ready to score against the included ground truth.
- **Dependency-light by default** — full test suite runs without GPUs or model downloads.

## Challenges

- Analog meter digits straddle rollovers and suffer glare/occlusion, making single-engine OCR brittle — hence the ensemble + LLM-verification design.
- Heavy OCR/LLM stacks are awkward in CI, motivating the mock-engine default so tests stay fast and deterministic.

## Future Work

- Wire `EnsembleOCR` + `LLMVerifier` into the default pipeline path behind a config flag.
- Run and **commit a real CER/WER benchmark** on the 927-image meter dataset, replacing the illustrative figures in `docs/`.
- Add PDF ingestion and multi-page table extraction end to end.

## License

[MIT](LICENSE) © 2025 Ejaz Fahil

## Conclusion

A thoughtfully layered OCR framework for industrial documents — pluggable multi-engine recognition, LLM-backed verification, real evaluation metrics, and a genuine meter-reading dataset — engineered so the demo runs anywhere while leaving a clear path to a production, fully benchmarked deployment.
