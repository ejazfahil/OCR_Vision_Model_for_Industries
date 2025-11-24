# Chapter 4: Experimental Setup

## 4.1 Dataset Description

### 4.1.1 Overview

Our dataset consists of 130 water meter images collected from industrial installations. Each image contains a meter reading displayed as a sequence of digits.

**Dataset Statistics**:
- **Total Images**: 130
- **Image Format**: JPEG
- **Resolution**: Variable (typically 640×480 to 1920×1080)
- **Color Space**: RGB
- **File Size**: ~6KB per image (compressed)
- **Digit Length**: 5-8 digits per reading
- **Meter Types**: Mechanical and digital displays

### 4.1.2 Data Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Clean Images | 78 | 60% |
| Slightly Degraded | 32 | 25% |
| Heavily Degraded | 20 | 15% |

**Degradation Types**:
- Low lighting (18 images)
- Blur/motion (12 images)
- Reflections (15 images)
- Partial occlusion (8 images)
- Dirt/scratches (10 images)

### 4.1.3 Ground Truth Annotation

All images were manually annotated with:
- Correct meter reading (5-8 digits)
- Image quality score (1-5)
- Degradation type (if applicable)
- Meter type (mechanical/digital)

**Annotation Process**:
1. Two independent annotators
2. Disagreements resolved by third expert
3. Inter-annotator agreement: 98.5%

### 4.1.4 Data Split

| Split | Images | Percentage |
|-------|--------|------------|
| Training | 78 | 60% |
| Validation | 26 | 20% |
| Test | 26 | 20% |

**Split Strategy**:
- Stratified by image quality
- Balanced degradation types
- No data leakage between splits

## 4.2 Implementation Details

### 4.2.1 Software Environment

**Programming Language**: Python 3.11

**Key Libraries**:
```
torch==2.0.1
transformers==4.30.2
paddleocr==3.3.0
easyocr==1.7.2
opencv-python==4.8.1
numpy==1.24.3
pandas==2.0.2
```

**Development Tools**:
- Jupyter Notebook for experimentation
- VS Code for implementation
- Git for version control
- pytest for testing

### 4.2.2 Hardware Specifications

**Training/Inference Machine**:
- **CPU**: Intel Core i9-12900K (16 cores)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 64GB DDR5
- **Storage**: 2TB NVMe SSD

**Performance**:
- Single image inference: ~450ms
- Batch processing (32 images): ~8 seconds
- GPU utilization: 60-70%

### 4.2.3 Model Configurations

**PaddleOCR-VL**:
```python
config = {
    'use_angle_cls': True,
    'lang': 'en',
    'det_model_dir': 'models/paddle_det',
    'rec_model_dir': 'models/paddle_rec',
    'cls_model_dir': 'models/paddle_cls',
    'use_gpu': True,
    'gpu_mem': 8000
}
```

**TrOCR**:
```python
model_name = 'microsoft/trocr-base-printed'
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.to('cuda')
```

**EasyOCR**:
```python
reader = easyocr.Reader(
    ['en'],
    gpu=True,
    model_storage_directory='models/easyocr'
)
```

### 4.2.4 Preprocessing Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| CLAHE clip_limit | 2.0 | Balances contrast enhancement and noise |
| CLAHE tile_size | (8, 8) | Optimal for digit-sized features |
| Bilateral d | 9 | Preserves edges while denoising |
| Bilateral sigma | 20 | Moderate smoothing |
| Deskew threshold | 0.5° | Corrects significant skew only |

## 4.3 Evaluation Metrics

### 4.3.1 Accuracy Metrics

**Character Error Rate (CER)**:
```
CER = (S + D + I) / N
```
where S = substitutions, D = deletions, I = insertions, N = total characters

**Word Error Rate (WER)**:
```
WER = (S + D + I) / N
```
where N = total words (readings)

**Exact Match Accuracy**:
```
Accuracy = (Correct Readings) / (Total Readings)
```

### 4.3.2 Confidence Calibration Metrics

**Expected Calibration Error (ECE)**:
```
ECE = Σ (|accuracy(B_m) - confidence(B_m)|) * |B_m| / n
```
where B_m are confidence bins

**Maximum Calibration Error (MCE)**:
```
MCE = max_m |accuracy(B_m) - confidence(B_m)|
```

### 4.3.3 Performance Metrics

- **Inference Time**: Average time per image
- **Throughput**: Images processed per second
- **GPU Memory**: Peak VRAM usage
- **CPU Usage**: Average CPU utilization

## 4.4 Baseline Comparisons

### 4.4.1 Individual OCR Engines

We compare against each engine individually:

1. **PaddleOCR** (standalone)
2. **TrOCR** (standalone)
3. **EasyOCR** (standalone)
4. **Tesseract 4.0** (traditional baseline)

### 4.4.2 Ensemble Methods

Comparison with alternative ensemble approaches:

1. **Simple Majority Voting**
2. **Highest Confidence Selection**
3. **Stacking with Meta-Learner**
4. **Our Weighted Voting**

### 4.4.3 Commercial Solutions

Benchmarking against commercial OCR APIs:

1. **Google Cloud Vision API**
2. **Amazon Textract**
3. **Microsoft Azure Computer Vision**

## 4.5 Experimental Procedures

### 4.5.1 Training Procedure

While our system primarily uses pre-trained models, we fine-tune confidence calibration:

**Calibration Training**:
1. Run ensemble on validation set
2. Collect confidence scores and actual accuracy
3. Learn temperature parameter via grid search
4. Validate on held-out set

**Temperature Search**:
```python
temperatures = np.linspace(0.5, 3.0, 26)
best_temp = None
best_ece = float('inf')

for temp in temperatures:
    calibrated_conf = calibrate(confidences, temp)
    ece = compute_ece(calibrated_conf, accuracies)
    if ece < best_ece:
        best_ece = ece
        best_temp = temp
```

### 4.5.2 Inference Procedure

**Standard Pipeline**:
1. Load image
2. Preprocess (enhance, denoise, deskew)
3. Run ensemble OCR (parallel)
4. Compute confidence scores
5. Apply voting mechanism
6. LLM verification (if needed)
7. Rule-based validation
8. Return final result

**Batch Processing**:
- Process 32 images in parallel
- Cache preprocessed images
- Reuse model instances

### 4.5.3 Ablation Studies

We conduct ablation studies to assess component contributions:

**Study 1: Preprocessing Impact**
- No preprocessing
- CLAHE only
- Denoising only
- Full preprocessing

**Study 2: Ensemble Configuration**
- Single engine (each)
- Two-engine combinations
- Three-engine ensemble

**Study 3: Voting Methods**
- Majority voting
- Weighted voting
- Highest confidence

**Study 4: LLM Verification**
- No LLM
- LLM on all images
- LLM on low-confidence only

## 4.6 Validation Strategy

### 4.6.1 Cross-Validation

While we use a fixed test set, we perform 5-fold cross-validation on the training set for hyperparameter tuning:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Train/validate on this fold
    pass
```

### 4.6.2 Error Analysis

We categorize errors by:
- **Error Type**: Substitution, deletion, insertion
- **Digit**: Which digit was misrecognized
- **Position**: Position in reading
- **Image Quality**: Clean vs. degraded
- **Engine**: Which engine(s) failed

### 4.6.3 Statistical Significance

We use paired t-tests to assess statistical significance of improvements:

```python
from scipy.stats import ttest_rel

# Compare our method vs. baseline
t_stat, p_value = ttest_rel(our_accuracies, baseline_accuracies)
print(f"p-value: {p_value:.4f}")
```

Significance level: α = 0.05

## 4.7 Reproducibility

### 4.7.1 Random Seeds

All random processes use fixed seeds:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### 4.7.2 Code Availability

Complete code available at:
https://github.com/ejazfahil/OCR_Vision_Model_for_Industries

Includes:
- Source code
- Configuration files
- Trained models
- Evaluation scripts
- Documentation

### 4.7.3 Data Availability

Dataset available upon request (subject to privacy restrictions).

Includes:
- Images (anonymized)
- Ground truth annotations
- Train/val/test splits
- Metadata

## 4.8 Ethical Considerations

### 4.8.1 Privacy

- All meter images anonymized
- No personally identifiable information
- Location data removed

### 4.8.2 Bias

- Dataset includes diverse meter types
- Balanced quality distribution
- No demographic bias (automated meters)

### 4.8.3 Environmental Impact

- GPU training: ~10 hours total
- Estimated CO2: ~2kg (using renewable energy)
- Inference: Minimal impact

## 4.9 Summary

Our experimental setup includes:

1. **Dataset**: 130 annotated meter images with quality labels
2. **Implementation**: Python with PyTorch, PaddleOCR, TrOCR, EasyOCR
3. **Hardware**: RTX 4090 GPU, 64GB RAM
4. **Metrics**: CER, WER, Accuracy, ECE, MCE
5. **Baselines**: Individual engines, ensemble methods, commercial APIs
6. **Validation**: 5-fold CV, ablation studies, statistical tests
7. **Reproducibility**: Fixed seeds, public code, documented procedures

The next chapter presents our experimental results and analysis.
