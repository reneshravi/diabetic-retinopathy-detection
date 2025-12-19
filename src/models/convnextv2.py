import torch
import torch.nn as nn
import timm

class ConvNeXtV2Model(nn.Module):
    """
    ConvNeXt V2 model wrapper for diabetic retinopathy classification.
    """
    
    def __init__(self, model_name='convnextv2_tiny', num_classes=5, 
                 pretrained=True, dropout=0.0):  
        """
        Args:
            model_name: name of ConvNeXt V2 variant
            num_classes: number of output classes
            pretrained: whether to use pretrained weights
            dropout: dropout rate for classification head (0.0 = no dropout)
        """
        super(ConvNeXtV2Model, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = dropout  
        
        # Create base model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # ADD DROPOUT LOGIC: Add dropout to classification head if specified
        if dropout > 0.0:
            # Get the original head
            original_head = self.model.head.fc
            
            # Replace with dropout + head
            self.model.head.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                original_head
            )
            
            print(f"Added dropout (p={dropout}) to classification head")
        
        print(f"Created {model_name} with {self.count_parameters():,} parameters")
        if pretrained:
            print("Using ImageNet pretrained weights")
        else:
            print("Training from scratch (random initialization)")
    
    def forward(self, x):
        return self.model(x)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier head"""
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        
        print("Backbone frozen. Only training classification head.")
    
    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
        
        print("All layers unfrozen.")


def get_model(model_name='convnextv2_tiny', num_classes=5, 
             pretrained=True, dropout=0.0): 
    """
    Factory function to create ConvNeXt V2 model.
    
    Available models:
    - convnextv2_atto
    - convnextv2_femto
    - convnextv2_pico
    - convnextv2_nano
    - convnextv2_tiny (recommended)
    - convnextv2_base
    - convnextv2_large
    
    Args:
        model_name: name of model variant
        num_classes: number of output classes
        pretrained: whether to use pretrained weights
        dropout: dropout rate (0.0-0.5, 0.0 = no dropout)
    
    Returns:
        ConvNeXtV2Model instance
    """
    model = ConvNeXtV2Model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout  
    )
    
    return model