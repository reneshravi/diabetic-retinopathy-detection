import sys
sys.path.append('.')

from src.models import get_model
from torchinfo import summary
import torch

# Test pre-trained model
print("Testing pre-trained model...")
model_pretrained = get_model('convnextv2_tiny', num_classes=5, pretrained=True)
summary(model_pretrained, input_size=(1, 3, 224, 224), 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3)

# See the actual architecture
print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
print(model)




# Test from-scratch model
# print("\nTesting from-scratch model...")
# model_scratch = get_model('convnextv2_tiny', num_classes=5, pretrained=False)

# Test forward pass
print("\nTesting forward pass...")
dummy_input = torch.randn(2, 3, 224, 224)
output = model_pretrained(dummy_input)
print(f"Output shape: {output.shape}")
print(f"✓ Model creation successful!")