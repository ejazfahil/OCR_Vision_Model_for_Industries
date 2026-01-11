import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ocr_engine import OCRResult, MockOCREngine

def test_ocr_result_to_dict():
    result = OCRResult("Hello", 0.95, (0, 0, 100, 20))
    d = result.to_dict()
    assert d["text"] == "Hello"
    assert d["confidence"] == 0.95
    assert isinstance(d["bbox"], list)

def test_mock_engine_returns_results():
    engine = MockOCREngine()
    results = engine.extract("fake_image.jpg")
    assert len(results) > 0
    assert all(isinstance(r, OCRResult) for r in results)

def test_ocr_result_confidence_valid():
    result = OCRResult("text", 0.88, (0, 0, 50, 20))
    assert 0.0 <= result.confidence <= 1.0

def test_mock_engine_has_engine_attr():
    engine = MockOCREngine(engine="easyocr")
    assert engine.engine == "easyocr"
