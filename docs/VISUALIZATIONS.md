# Visual Documentation

This directory contains all visualizations and diagrams for the OCR Vision Model project.

## Architecture Diagrams

### System Architecture
![System Architecture](images/system_architecture_1763942231092.png)

Complete system architecture showing the flow from input image through preprocessing, ensemble OCR, confidence scoring, LLM verification, and final output.

## Performance Visualizations

### Overall Performance Comparison
![Performance Comparison](images/performance_comparison_1763942245522.png)

Comparison of accuracy across different OCR methods, demonstrating our ensemble + LLM approach achieves 97.8% accuracy.

### Performance by Image Quality
![Quality Performance](images/quality_performance_1763942263459.png)

Accuracy breakdown by image quality level (clean, slightly degraded, heavily degraded), showing consistent performance across all conditions.

### Ablation Study Results
![Ablation Study](images/ablation_study_1763942279429.png)

Component contribution analysis demonstrating the value of each system component.

## Technical Metrics

### Confidence Calibration
![Confidence Calibration](images/confidence_calibration_1763942296737.png)

Reliability diagram showing excellent calibration (ECE=0.042) between predicted confidence and actual accuracy.

### Processing Time Distribution
![Processing Time](images/processing_time_1763942311971.png)

Breakdown of processing time across system components (total: 450ms per image).

## Usage in Documentation

These images are referenced in:
- Main README.md
- Thesis chapters (methodology, results)
- API documentation
- Presentations

## Image Details

| Image | Size | Format | Purpose |
|-------|------|--------|---------|
| system_architecture | ~500KB | PNG | Architecture overview |
| performance_comparison | ~300KB | PNG | Results visualization |
| quality_performance | ~350KB | PNG | Quality analysis |
| ablation_study | ~300KB | PNG | Component analysis |
| confidence_calibration | ~250KB | PNG | Calibration metrics |
| processing_time | ~200KB | PNG | Performance breakdown |

All images are high-resolution PNG format suitable for presentations and publications.
