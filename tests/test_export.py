import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.export import to_json, to_csv

def test_to_json_creates_file():
    data = [{"file": "a.pdf", "confidence": 0.9}]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    to_json(data, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded[0]["file"] == "a.pdf"

def test_to_csv_creates_file():
    data = [{"file": "b.pdf", "confidence": 0.85}]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    to_csv(data, path)
    with open(path) as f:
        content = f.read()
    assert "b.pdf" in content

def test_to_json_empty_list():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    to_json([], path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == []
