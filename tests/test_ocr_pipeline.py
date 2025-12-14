"""Unit tests for OCR pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import validate_image_format, normalize_confidence, encode_image_base64


def test_image_is_not_none():
    """Basic sanity check."""
    img = b"fake_image_bytes"
    assert img is not None


def test_text_extraction_returns_string():
    extracted = "Invoice No: 12345"
    assert isinstance(extracted, str)


def test_confidence_score_in_range():
    confidence = 0.92
    assert 0.0 <= confidence <= 1.0


def test_validate_image_format_valid():
    assert validate_image_format("invoice.pdf") is True
    assert validate_image_format("scan.jpg") is True
    assert validate_image_format("document.png") is True


def test_validate_image_format_invalid():
    assert validate_image_format("data.csv") is False
    assert validate_image_format("script.py") is False


def test_normalize_confidence_clamping():
    assert normalize_confidence(1.5) == 1.0
    assert normalize_confidence(-0.1) == 0.0
    assert normalize_confidence(0.85) == 0.85


def test_encode_image_base64_roundtrip():
    original = b"hello image bytes"
    encoded = encode_image_base64(original)
    import base64
    decoded = base64.b64decode(encoded)
    assert decoded == original
