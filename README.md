# Diabetic Retinopathy Detection with ConvNeXt V2

A deep learning system for automated detection and severity classification of diabetic retinopathy from retinal fundus images. This project implements both transfer learning and from-scratch training approaches using the ConvNeXt V2 architecture.

## Overview

Diabetic retinopathy is a leading cause of blindness worldwide. This project uses state-of-the-art deep learning to classify retinal images into 5 severity levels:
- **0**: No DR
- **1**: Mild DR
- **2**: Moderate DR
- **3**: Severe DR
- **4**: Proliferative DR

The system achieves strong performance using transfer learning with pre-trained ConvNeXt V2 models, demonstrating the effectiveness of modern convolutional architectures for medical imaging tasks.

## Dataset

**APTOS 2019 Blindness Detection**
- Source: [Kaggle Competition](https://www.kaggle.com/c/aptos2019-blindness-detection)
- Training images: 3,662 retinal fundus photographs
- 5-class ordinal classification task
- Significant class imbalance with predominance of class 0 (No DR)

### Class Distribution
```
Class 0 (No DR):          1,805 images (49.3%)
Class 1 (Mild):             370 images (10.1%)
Class 2 (Moderate):         999 images (27.3%)
Class 3 (Severe):           193 images (5.3%)
Class 4 (Proliferative):    295 images (8.1%)
```

## Architecture

### ConvNeXt V2 Base
- Modern pure convolutional architecture
- Improved over Vision Transformers for smaller datasets
- Global Response Normalization (GRN) for better feature learning
- Modified classification head for 5-class output

### Why ConvNeXt V2?
- Excellent transfer learning capabilities from ImageNet
- Better performance than ViT on medical imaging datasets
- More parameter-efficient than transformer alternatives
- Strong feature extraction for high-resolution retinal images

## Data Pipeline

### Preprocessing Steps
1. **Circle Cropping**: Removes black borders and focuses on retinal area
2. **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for improved vessel visibility
3. **Resizing**: 384×384 pixels for model input
4. **Normalization**: ImageNet statistics for transfer learning compatibility

### Data Augmentation
- Random horizontal/vertical flips
- Random rotation (±15 degrees)
- Color jittering (brightness, contrast, saturation)
- Random affine transformations
- Applied only during training

## Implementations

### 1. Pre-trained Model (Primary)
Fine-tunes ConvNeXt V2 model pre-trained on ImageNet-1K.

**Training Strategy:**
- **Phase 1**: Freeze backbone, train head only (10 epochs)
- **Phase 2**: Unfreeze all layers, end-to-end fine-tuning (40 epochs)
- Learning rate: 3e-5 (backbone), 1e-4 (head)
- AdamW optimizer with cosine annealing schedule

**Current Performance:**
- Validation QWK: 0.87+ by epoch 10
- Strong generalization without overfitting
- Rapid convergence in frozen phase

### 2. From-Scratch Model (Experimental - TBD)
Trains ConvNeXt V2 with random initialization for comparison.

**Purpose:**
- Understand fundamental learning dynamics
- Quantify transfer learning benefits
- Explore architecture capacity on medical data

## Evaluation Metrics

### Primary Metric: Quadratic Weighted Kappa (QWK)
- Measures agreement accounting for severity ordering
- Penalizes distant misclassifications more heavily
- Range: -1 (worst) to 1 (perfect)
- Clinical relevance: distinguishes minor vs severe errors

### Secondary Metrics
- **Accuracy**: Overall classification correctness
- **Per-class Precision/Recall**: Performance across severity levels
- **Confusion Matrix**: Detailed error analysis

## Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- timm (PyTorch Image Models)
- OpenCV
- scikit-learn
- PyYAML
- tqdm

### Installation
```bash
# Clone repository
git clone <repository-url>
cd diabetic-retinopathy-detection

# Install dependencies
pip install -r requirements.txt

# Download APTOS 2019 dataset
# Place in data/aptos2019/ with structure:
# data/aptos2019/
#   ├── train_images/
#   ├── test_images/
#   └── train.csv
```

## Usage

### Training

**Pre-trained Model (Recommended):**
```bash
python scripts/train_pretrained.py
```

**From-Scratch Model:**
```bash
python scripts/train_scratch.py
```

**Custom Configuration:**
```bash
# Edit config files in configs/
python scripts/train_pretrained.py --config configs/custom_config.yaml
```

### Evaluation

```bash
# Evaluate specific checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --test
```

### Training Configuration

Key parameters in `configs/pretrained_config.yaml`:
```yaml
model:
  architecture: "convnextv2_base.fcmae_ft_in1k"
  num_classes: 5
  pretrained: true

training:
  batch_size: 16
  num_epochs_frozen: 10
  num_epochs_unfrozen: 40
  lr_head: 0.0001
  lr_backbone: 0.00003
  weight_decay: 0.01

data:
  image_size: 384
  train_split: 0.85
  use_class_weights: true
```

## Project Structure

```
diabetic-retinopathy-detection/
├── configs/
│   ├── pretrained_config.yaml
│   └── scratch_config.yaml
├── data/
│   └── aptos2019/
│       ├── train_images/
│       ├── test_images/
│       └── train.csv
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset implementation
│   │   └── preprocessing.py      # Image preprocessing
│   ├── models/
│   │   ├── convnext.py          # Model architecture
│   │   └── utils.py             # Model utilities
│   └── utils/
│       ├── metrics.py            # Evaluation metrics
│       └── training.py           # Training utilities
├── scripts/
│   ├── train_pretrained.py       # Pre-trained training
│   ├── train_scratch.py          # From-scratch training
│   └── evaluate.py               # Model evaluation
├── checkpoints/                  # Saved model weights
├── logs/                         # Training logs
├── requirements.txt
└── README.md
```

## Results

### Pre-trained ConvNeXt V2

| Metric | Frozen Phase (10 epochs) | Fine-tuned (50 epochs) |
|--------|-------------------------|------------------------|
| Validation QWK | 0.87+ | TBD |
| Validation Accuracy | TBD | TBD |
| Training Time (CPU) | ~3-4 hours | TBD |

**Observations:**
- Rapid convergence during frozen backbone training
- Strong performance with transfer learning
- No signs of overfitting with proper regularization
- Class imbalance handled via weighted loss function

### From-Scratch Model
*Results pending training completion*

### Comparison Analysis
*Detailed comparison between pre-trained and from-scratch approaches coming soon*

## Key Learnings

1. **Transfer Learning Effectiveness**: Pre-trained models provide significant advantage for medical imaging with limited data
2. **Architecture Choice**: ConvNeXt V2 outperforms Vision Transformers on smaller medical datasets
3. **Class Imbalance**: Weighted loss functions critical for handling skewed distributions
4. **Medical Preprocessing**: Circle cropping and CLAHE enhancement improve model performance
5. **Evaluation Metrics**: QWK more appropriate than accuracy for ordinal clinical tasks

## Future Work

- [ ] Complete from-scratch training and comparison analysis
- [ ] Implement ensemble methods with multiple architectures
- [ ] Add external validation on EyePACS or Messidor datasets
- [ ] Explore test-time augmentation for improved predictions
- [ ] Investigate attention visualization for clinical interpretability
- [ ] Optimize for production deployment (quantization, ONNX export)

## References

1. **ConvNeXt V2**: Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (2023)
2. **Dataset**: APTOS 2019 Blindness Detection Competition, Kaggle
3. **Medical Imaging**: Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy" (2016)

## License

This project is for educational and research purposes. Please refer to the APTOS 2019 dataset terms for data usage restrictions.

## Acknowledgments

- APTOS India for providing the dataset
- Kaggle for hosting the competition
- Ross Wightman for the `timm` library
- PyTorch team for the deep learning framework
