"""
CFPModule (Context Fusion with Pyramid Dilated Convolutions)

Ported from ColonFormer to provide multi-scale context aggregation.
Uses 4 parallel branches with different dilation rates to capture
features at multiple receptive fields.

Architecture:
- Input: [B, C, H, W]
- 4 parallel branches with dilations: [1, d/4+1, d/2+1, d+1]
- Each branch: 3 cascade conv layers (depthwise grouped)
- Progressive fusion of branches
- Residual connection
- Output: [B, C, H, W] (same as input)

Key differences from ColonMamba's CFP:
- CFPModule: Multi-scale context via dilated convs (single feature map)
- CFP (in decoder.py): Cross-scale fusion via FPN (two feature maps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CFPModule(nn.Module):
    """
    Context Fusion Pyramid Module with multi-scale dilated convolutions.
    
    Args:
        nIn (int): Number of input channels
        d (int): Base dilation rate (default: 8)
        KSize (int): Kernel size for initial reduction (default: 3)
        dkSize (int): Kernel size for dilated convs (default: 3)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(self, nIn, d=8, KSize=3, dkSize=3, dropout=0.1):
        super(CFPModule, self).__init__()
        
        self.nIn = nIn
        self.d = d
        self.dropout = dropout
        
        # Pre-normalization
        self.bn_relu_1 = nn.Sequential(
            nn.BatchNorm2d(nIn, eps=1e-3),
            nn.PReLU(nIn)
        )
        
        # Post-normalization
        self.bn_relu_2 = nn.Sequential(
            nn.BatchNorm2d(nIn, eps=1e-3),
            nn.PReLU(nIn)
        )
        
        # Reduce channels: nIn → nIn/4
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(nIn, nIn // 4, KSize, 1, padding=1, bias=False),
            nn.BatchNorm2d(nIn // 4, eps=1e-3),
            nn.PReLU(nIn // 4),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()  # ← ADD
        )
        
        # ===== Branch 1: dilation = 1 =====
        self.dconv_1_1 = self._make_conv(
            nIn // 4, nIn // 16, dkSize, dilation=1, padding=1
        )
        self.dconv_1_2 = self._make_conv(
            nIn // 16, nIn // 16, dkSize, dilation=1, padding=1
        )
        self.dconv_1_3 = self._make_conv(
            nIn // 16, nIn // 8, dkSize, dilation=1, padding=1
        )
        
        # ===== Branch 2: dilation = d/4 + 1 =====
        d2 = int(d / 4 + 1)
        self.dconv_2_1 = self._make_conv(
            nIn // 4, nIn // 16, dkSize, dilation=d2, padding=d2
        )
        self.dconv_2_2 = self._make_conv(
            nIn // 16, nIn // 16, dkSize, dilation=d2, padding=d2
        )
        self.dconv_2_3 = self._make_conv(
            nIn // 16, nIn // 8, dkSize, dilation=d2, padding=d2
        )
        
        # ===== Branch 3: dilation = d/2 + 1 =====
        d3 = int(d / 2 + 1)
        self.dconv_3_1 = self._make_conv(
            nIn // 4, nIn // 16, dkSize, dilation=d3, padding=d3
        )
        self.dconv_3_2 = self._make_conv(
            nIn // 16, nIn // 16, dkSize, dilation=d3, padding=d3
        )
        self.dconv_3_3 = self._make_conv(
            nIn // 16, nIn // 8, dkSize, dilation=d3, padding=d3
        )
        
        # ===== Branch 4: dilation = d + 1 =====
        d4 = d + 1
        self.dconv_4_1 = self._make_conv(
            nIn // 4, nIn // 16, dkSize, dilation=d4, padding=d4
        )
        self.dconv_4_2 = self._make_conv(
            nIn // 16, nIn // 16, dkSize, dilation=d4, padding=d4
        )
        self.dconv_4_3 = self._make_conv(
            nIn // 16, nIn // 8, dkSize, dilation=d4, padding=d4
        )
        
        # Final 1x1 conv to restore channels: nIn → nIn
        self.conv1x1 = nn.Conv2d(nIn, nIn, 1, 1, padding=0, bias=False)
    
    def _make_conv(self, in_channels, out_channels, kernel_size, dilation, padding):
        """
        Create grouped convolution with dynamic groups using GCD.
        Supports dual-mode architecture with varying channels.
        """
        # Use GCD to find valid groups
        groups = math.gcd(in_channels, out_channels)
        
        return nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=(kernel_size, kernel_size),
                stride=1,
                padding=(padding, padding),
                dilation=(dilation, dilation),
                groups=groups,  # ← FIX: Dynamic groups
                bias=False
            ),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.PReLU(out_channels)
        )
    
    def forward(self, input):
        """
        Forward pass through CFPModule.
        
        Args:
            input (Tensor): Input features [B, C, H, W]
        
        Returns:
            Tensor: Context-enriched features [B, C, H, W]
        """
        # Save input for residual connection
        identity = input
        
        # ===== Pre-processing =====
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)  # [B, C/4, H, W]
        
        # ===== Branch 1 (dilation=1) =====
        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)
        
        # ===== Branch 2 (dilation=d/4+1) =====
        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)
        
        # ===== Branch 3 (dilation=d/2+1) =====
        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)
        
        # ===== Branch 4 (dilation=d+1) =====
        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)
        
        # ===== Concatenate branch outputs =====
        # Each branch produces 3 outputs: C/16 + C/16 + C/8 = C/4
        output_1 = torch.cat([o1_1, o1_2, o1_3], dim=1)  # [B, C/4, H, W]
        output_2 = torch.cat([o2_1, o2_2, o2_3], dim=1)  # [B, C/4, H, W]
        output_3 = torch.cat([o3_1, o3_2, o3_3], dim=1)  # [B, C/4, H, W]
        output_4 = torch.cat([o4_1, o4_2, o4_3], dim=1)  # [B, C/4, H, W]
        
        # ===== Progressive fusion =====
        # Accumulate context from different scales
        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        
        # Concatenate all accumulated features: 4 * C/4 = C
        output = torch.cat([ad1, ad2, ad3, ad4], dim=1)  # [B, C, H, W]
        
        # ===== Post-processing =====
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)  # [B, C, H, W]

        # ===== Dropout =====
        if self.dropout > 0:
            output = F.dropout2d(output, p=self.dropout, training=self.training)
        
        # ===== Residual connection =====
        output = output + identity
        
        return output
    
    def extra_repr(self):
        """Extra representation for printing"""
        return f'nIn={self.nIn}, d={self.d}'


if __name__ == '__main__':
    # Unit test
    print("="*60)
    print("Testing CFPModule")
    print("="*60)
    
    # Test with different channel sizes (matching encoder outputs)
    test_configs = [
        (128, 44, 44),   # F2: 1/8 scale
        (384, 22, 22),   # F3: 1/16 scale
        (768, 11, 11),   # F4: 1/32 scale
    ]
    
    for channels, h, w in test_configs:
        print(f"\nTesting CFPModule with {channels} channels, {h}x{w}...")
        
        # Create module
        cfp = CFPModule(nIn=channels, d=8)
        num_params = sum(p.numel() for p in cfp.parameters())
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, channels, h, w)
        print(f"  Input shape: {x.shape}")
        
        output = cfp(x)
        print(f"  Output shape: {output.shape}")
        
        # Verify shape
        assert output.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {output.shape}"
        print(f"  ✓ Shape verified!")
        
        # Test gradient flow
        loss = output.mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in cfp.parameters())
        assert has_grad, "No gradients!"
        print(f"  ✓ Gradients flow correctly!")
    
    print("\n" + "="*60)
    print("✓ All CFPModule tests passed!")
    print("="*60)
