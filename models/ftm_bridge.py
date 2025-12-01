"""
Feature Transition Module (FTM) - Bridge between ResNet and VMamba

This module acts as a "semantic projection" layer, transitioning from CNN 
low-level features (128 channels) to Mamba high-level semantic space (384 channels).

Design rationale:
- Conv1x1: Channel projection without spatial information loss
- BatchNorm: Normalizes CNN-style features (ResNet uses BN)
- GELU: Smooth activation compatible with Mamba (better than ReLU)
- NO Residual: Dimension mismatch (128 → 384) makes residual impossible
"""

import torch
import torch.nn as nn


class FTM_Bridge(nn.Module):
    """
    Feature Transition Module - Semantic bridge between CNN and Mamba stages.
    
    Args:
        in_channels (int): Input channels from ResNet Stage 2 (default: 128)
        out_channels (int): Output channels for VMamba Stage 3 (default: 384)
    
    Input shape: [B, 128, H/8, W/8]
    Output shape: [B, 384, H/8, W/8]
    
    Technical Note:
    This layer projects low-level morphological features (edges, textures from CNN)
    into a higher-dimensional semantic space where global context modeling (Mamba)
    can better capture long-range dependencies in polyp segmentation.
    """
    
    def __init__(self, in_channels=128, out_channels=384):
        super(FTM_Bridge, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False  # No bias before BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.act = nn.GELU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize Conv weights with Kaiming normal (good for GELU)"""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through FTM Bridge.
        
        Args:
            x (Tensor): Input features from ResNet Layer2, shape [B, 128, H/8, W/8]
        
        Returns:
            Tensor: Projected features ready for VMamba, shape [B, 384, H/8, W/8]
        """
        # Project channels: 128 → 384
        x = self.conv(x)        # [B, 384, H/8, W/8]
        
        # Normalize features
        x = self.bn(x)          # [B, 384, H/8, W/8]
        
        # Apply smooth activation
        x = self.act(x)         # [B, 384, H/8, W/8]
        
        return x
    
    def extra_repr(self):
        """Extra representation for printing"""
        return f'in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}'


if __name__ == '__main__':
    # Unit test
    print("Testing FTM_Bridge module...")
    
    # Create module
    ftm = FTM_Bridge(in_channels=128, out_channels=384)
    print(f"FTM_Bridge: {ftm}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 128, 44, 44)  # Simulating H=W=352, scale 1/8
    print(f"Input shape: {x.shape}")
    
    output = ftm(x)
    print(f"Output shape: {output.shape}")
    
    # Verify shape
    assert output.shape == (batch_size, 384, 44, 44), "Output shape mismatch!"
    print("✓ FTM_Bridge unit test passed!")
    
    # Count parameters
    num_params = sum(p.numel() for p in ftm.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
