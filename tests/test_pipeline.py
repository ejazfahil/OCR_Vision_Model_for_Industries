import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import OCRPipeline
import pytest

def test_pipeline_rejects_invalid_format():
    pipeline = OCRPipeline()
    with pytest.raises(ValueError):
        pipeline.process("data.csv")

def test_pipeline_returns_dict_with_required_keys():
    pipeline = OCRPipeline()
    result = pipeline.process("invoice.pdf")
    assert "file" in result
    assert "confidence" in result
    assert "fields" in result
    assert "raw_text" in result

def test_pipeline_confidence_in_range():
    pipeline = OCRPipeline()
    result = pipeline.process("scan.jpg")
    assert 0.0 <= result["confidence"] <= 1.0

def test_pipeline_fields_has_expected_keys():
    pipeline = OCRPipeline()
    result = pipeline.process("invoice.pdf")
    assert "invoice_number" in result["fields"]
    assert "date" in result["fields"]
    assert "amount" in result["fields"]
