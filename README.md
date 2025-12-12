# Diabetic Retinopathy Detection with ConvNeXt V2

Classification of diabetic retinopathy severity using ConvNeXt V2 architecture.

## Dataset
APTOS 2019 Blindness Detection

## Implementations
1. **Pre-trained**: Fine-tuning ImageNet pre-trained model
2. **From-Scratch**: Training with random initialization

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train pre-trained model
python scripts/train_pretrained.py

# Train from-scratch model
python scripts/train_scratch.py

# Evaluate model
python scripts/evaluate.py --checkpoint path/to/model.pth
```

## Results
