import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def get_cloud_model(pretrained=True, num_channels=4):
    """
    Returns a modified MobileNetV3-Small for binary cloud classification.
    
    Args:
        pretrained (bool): If True, loads ImageNet weights for RGB channels.
        num_channels (int): Number of input channels (e.g. 4 for RGB+NIR).
    """
    if pretrained:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
    else:
        model = mobilenet_v3_small(weights=None)
        
    # 1. Adapt input channels if necessary
    if num_channels != 3:
        # Original first layer: Conv2dNormActivation(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_conv = model.features[0][0]
        
        # Create new conv layer with num_channels
        new_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        if pretrained:
            # Copy weights from RGB channels
            with torch.no_grad():
                # Shape: [out_channels, in_channels, H, W]
                new_conv.weight[:, :3, :, :] = original_conv.weight.clone()
                # For the 4th channel (NIR), we can initialize it with the mean of RGB weights
                # or copy the Red channel weights (index 0). Let's copy Red channel.
                new_conv.weight[:, 3:, :, :] = original_conv.weight[:, 0:1, :, :].clone()
                
        # Replace the first layer
        model.features[0][0] = new_conv
        
    # 2. Adapt the classifier for binary classification
    # Original classifier:
    # (0): Linear(in_features=576, out_features=1024, bias=True)
    # (1): Hardswish()
    # (2): Dropout(p=0.2, inplace=True)
    # (3): Linear(in_features=1024, out_features=1000, bias=True)
    
    in_features = model.classifier[3].in_features
    # Replace the last linear layer to output 1 logit
    model.classifier[3] = nn.Linear(in_features, 1)
    
    return model

if __name__ == '__main__':
    # Simple test
    model = get_cloud_model(pretrained=True, num_channels=4)
    dummy_input = torch.randn(1, 4, 256, 256)
    output = model(dummy_input)
    print("Output shape:", output.shape) # Expected: (1, 1)
