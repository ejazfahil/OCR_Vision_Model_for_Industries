"""Export OCR results to JSON and CSV formats."""
import json, csv
from typing import List, Dict, Any

def to_json(results: List[Dict[str, Any]], filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

def to_csv(results: List[Dict[str, Any]], filepath: str) -> None:
    if not results:
        return
    keys = ["file", "confidence"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
