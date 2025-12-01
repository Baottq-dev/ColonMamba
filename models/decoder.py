"""
PPD (Parallel Partial Decoder) with Mamba Reverse Refinement

Based on ColonFormer's decoder but adapted for Hybrid Res-VMamba encoder:
- Input channels: [64, 128, 384, 768] (modified from original [64, 128, 320, 512])
- MRR refinement at scales 2, 3, 4 (replaces RA-RA)
- Deep supervision with auxiliary outputs at 3 scales

Key Components:
- CFP (Context Fusion Module): Multi-scale context aggregation  
- PPM (Pyramid Pooling Module): Global context on deepest features
- MRR: Boundary refinement with Mamba cross-scanning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mrr import MambaReverseRefinement
from .cfp_module import CFPModule


class PPM(nn.Module):
    """
    Pyramid Pooling Module for capturing global context.
    Applies pooling at multiple scales and fuses them.
    """
    
    def __init__(self, in_channels, out_channels=512, pool_sizes=[1, 2, 3, 6]):
        super(PPM, self).__init__()
        
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create pooling branches
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(out_channels // len(pool_sizes), eps=1e-3),
                nn.ReLU(inplace=True)
            )
            for pool_size in pool_sizes
        ])
        
        # Bottleneck to combine all branches
        total_channels = in_channels + out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)  # Fixed typo: inpace -> inplace
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels, H, W]
        """
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        
        for stage in self.stages:
            # Pool and process
            pooled = stage(x)
            # Upsample back to original size
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(upsampled)
        
        # Concatenate all pyramid levels
        output = torch.cat(pyramids, dim=1)
        
        # Fuse with bottleneck
        output = self.bottleneck(output)
        
        return output


class CFP(nn.Module):
    """
    Context Fusion Module - Fuses features from different encoder scales.
    Similar to FPN but with additional context modeling.
    """
    
    def __init__(self, in_channels_low, in_channels_high, out_channels):
        super(CFP, self).__init__()
        
        # Process low-resolution (high-level) features
        self.conv_low = nn.Sequential(
            nn.Conv2d(in_channels_low, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Process high-resolution (low-level) features
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels_high, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Fusion convolution
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feat_low, feat_high):
        """
        Args:
            feat_low: Low-res features [B, C_low, H, W]
            feat_high: High-res features [B, C_high, 2H, 2W]
        Returns:
            Fused features [B, out_channels, 2H, 2W]
        """
        # Process low-res features
        feat_low = self.conv_low(feat_low)
        
        # Upsample low-res to match high-res
        h, w = feat_high.size(2), feat_high.size(3)
        feat_low_up = F.interpolate(feat_low, size=(h, w), mode='bilinear', align_corners=True)
        
        # Process high-res features
        feat_high = self.conv_high(feat_high)
        
        # Concatenate and fuse
        feat_cat = torch.cat([feat_low_up, feat_high], dim=1)
        output = self.conv_fusion(feat_cat)
        
        return output


class PPDDecoder(nn.Module):
    """
    Parallel Partial Decoder with Mamba Reverse Refinement.
    
    Modified from ColonFormer to work with Hybrid Res-VMamba encoder:
    - Input channels: [64, 128, 384, 768] at scales [1/4, 1/8, 1/16, 1/32]
    - **NEW**: CFPModule enhancement for encoder features (multi-scale context)
    - Applies MRR at scales 2, 3, 4
    - Produces deep supervision outputs at 3 scales
    
    Args:
        in_channels (list): Encoder output channels [64, 128, 384, 768]
        num_classes (int): Number of output classes (default: 1 for binary segmentation)
        use_mrr (bool): Use Mamba Reverse Refinement (default: True)
        use_cfp_enhance (bool): Use CFPModule to enhance encoder features (default: True)
    """
    # NOTE: Multi-scale architecture - preserves encoder channels throughout decoder
    # Unlike unified approach, each scale maintains its original channel dimension
    
    def __init__(
        self, 
        in_channels=[64, 128, 384, 768],
        num_classes=1,
        use_mrr=True,
        use_cfp_enhance=True,
    ):
        super(PPDDecoder, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.use_mrr = use_mrr
        self.use_cfp_enhance = use_cfp_enhance
        
        # Unpack encoder channel dimensions
        c1, c2, c3, c4 = in_channels  # [64, 128, 384, 768]
        # Multi-scale: each decoder stage keeps its corresponding encoder channels
        
        # ============ CFPModule Enhancement (NEW) ============
        # Apply multi-scale dilated convolutions to encoder features
        # This matches ColonFormer workflow where CFP enriches features BEFORE decode
        if use_cfp_enhance:
            self.cfp_enhance_2 = CFPModule(c2, d=8)  # Enhance F2 (128 channels)
            self.cfp_enhance_3 = CFPModule(c3, d=8)  # Enhance F3 (384 channels)
            self.cfp_enhance_4 = CFPModule(c4, d=8)  # Enhance F4 (768 channels)
        
        # ============ Deepest Level Processing (F4) ============
        # Apply PPM on F4 (1/32 scale, 768 channels)
        # Multi-scale: Keep c4 channels (768) instead of reducing to 256
        self.ppm = PPM(c4, out_channels=c4, pool_sizes=[1, 2, 3, 6])
        
        # ============ Context Fusion Modules ============
        # Multi-scale: Output channels match the high-res feature channels
        # CFP for F3 (1/16) - fuse with upsampled F4, output c3 (384)
        self.cfp3 = CFP(in_channels_low=c4, in_channels_high=c3, out_channels=c3)
        
        # CFP for F2 (1/8) - fuse with upsampled result from F3, output c2 (128)
        self.cfp2 = CFP(in_channels_low=c3, in_channels_high=c2, out_channels=c2)
        
        # CFP for F1 (1/4) - fuse with upsampled result from F2, output c1 (64)
        self.cfp1 = CFP(in_channels_low=c2, in_channels_high=c1, out_channels=c1)
        
        # ============ Mamba Reverse Refinement (MRR) ============
        # Multi-scale: Each MRR matches its corresponding scale's channels
        if use_mrr:
            # MRR at scale 4 (1/32) - 768 channels
            self.mrr4 = MambaReverseRefinement(dim=c4, d_state=16, d_conv=4, expand=2)
            
            # MRR at scale 3 (1/16) - 384 channels
            self.mrr3 = MambaReverseRefinement(dim=c3, d_state=16, d_conv=4, expand=2)
            
            # MRR at scale 2 (1/8) - 128 channels
            self.mrr2 = MambaReverseRefinement(dim=c2, d_state=16, d_conv=4, expand=2)
        
        # ============ Prediction Heads (Deep Supervision) ============
        # Multi-scale: Each head receives features with different channel counts
        # Auxiliary head for scale 4 (1/32) - 768 channels
        self.aux_head_4 = nn.Conv2d(c4, num_classes, 1)
        
        # Auxiliary head for scale 3 (1/16) - 384 channels
        self.aux_head_3 = nn.Conv2d(c3, num_classes, 1)
        
        # Auxiliary head for scale 2 (1/8) - 128 channels
        self.aux_head_2 = nn.Conv2d(c2, num_classes, 1)
        
        # Main prediction head (full resolution) - 64 channels
        self.main_head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1 // 2, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, num_classes, 1)
        )
    
    def forward(self, features):
        """
        Forward pass through PPD decoder.
        
        Args:
            features (list): Multi-scale features from encoder [F1, F2, F3, F4]
                F1: [B, 64, H/4, W/4]
                F2: [B, 128, H/8, W/8]
                F3: [B, 384, H/16, W/16]
                F4: [B, 768, H/32, W/32]
        
        Returns:
            dict: Predictions at multiple scales
                'main': [B, num_classes, H, W] - main output
                'aux_1_8': [B, num_classes, H/8, W/8] - auxiliary output
                'aux_1_16': [B, num_classes, H/16, W/16] - auxiliary output
                'aux_1_32': [B, num_classes, H/32, W/32] - auxiliary output
        """
        F1, F2, F3, F4 = features
        B, _, H, W = F1.shape
        
        # ============ CFPModule Enhancement (NEW) ============
        # Enrich encoder features with multi-scale context BEFORE decoding
        # This matches ColonFormer workflow
        if self.use_cfp_enhance:
            F2 = self.cfp_enhance_2(F2)  # [B, 128, H/8, W/8] - enhanced
            F3 = self.cfp_enhance_3(F3)  # [B, 384, H/16, W/16] - enhanced
            F4 = self.cfp_enhance_4(F4)  # [B, 768, H/32, W/32] - enhanced
        
        # ============ Decode from deepest level ============
        # Apply PPM on enhanced F4
        D4 = self.ppm(F4)  # [B, 768, H/32, W/32] - Multi-scale: keeps c4 channels
        
        # Auxiliary prediction at 1/32 scale
        aux_pred_4 = self.aux_head_4(D4)  # [B, num_classes, H/32, W/32]
        
        # Apply MRR if enabled
        if self.use_mrr:
            D4 = self.mrr4(D4, aux_pred_4)  # [B, 768, H/32, W/32]
        
        # ============ Fuse with F3 (1/16 scale) ============
        D3 = self.cfp3(D4, F3)  # [B, 384, H/16, W/16] - Multi-scale: outputs c3
        
        # Auxiliary prediction at 1/16 scale
        aux_pred_3 = self.aux_head_3(D3)  # [B, num_classes, H/16, W/16]
        
        # Apply MRR
        if self.use_mrr:
            D3 = self.mrr3(D3, aux_pred_3)  # [B, 384, H/16, W/16]
        
        # ============ Fuse with F2 (1/8 scale) ============
        D2 = self.cfp2(D3, F2)  # [B, 128, H/8, W/8] - Multi-scale: outputs c2
        
        # Auxiliary prediction at 1/8 scale
        aux_pred_2 = self.aux_head_2(D2)  # [B, num_classes, H/8, W/8]
        
        # Apply MRR
        if self.use_mrr:
            D2 = self.mrr2(D2, aux_pred_2)  # [B, 128, H/8, W/8]
        
        # ============ Fuse with F1 (1/4 scale) ============
        D1 = self.cfp1(D2, F1)  # [B, 64, H/4, W/4] - Multi-scale: outputs c1
        
        # ============ Final upsampling and prediction ============
        # Upsample to full resolution
        D_final = F.interpolate(D1, size=(H, W), mode='bilinear', align_corners=True)
        
        # Main prediction
        main_pred = self.main_head(D_final)  # [B, num_classes, H, W]
        
        # Return all predictions for deep supervision
        return {
            'main': main_pred,
            'aux_1_8': aux_pred_2,
            'aux_1_16': aux_pred_3,
            'aux_1_32': aux_pred_4,
        }


if __name__ == '__main__':
    # Unit test
    print("Testing PPDDecoder...")
    
    # Create decoder
    decoder = PPDDecoder(
        in_channels=[64, 128, 384, 768],
        num_classes=1,
        decoder_channels=256,
        use_mrr=True,
    )
    
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Create dummy encoder features
    batch_size = 2
    H, W = 352, 352
    
    features = [
        torch.randn(batch_size, 64, H//4, W//4),     # F1: 1/4 scale
        torch.randn(batch_size, 128, H//8, W//8),    # F2: 1/8 scale
        torch.randn(batch_size, 384, H//16, W//16),  # F3: 1/16 scale
        torch.randn(batch_size, 768, H//32, W//32),  # F4: 1/32 scale
    ]
    
    print("\nInput features:")
    for i, feat in enumerate(features, 1):
        print(f"  F{i}: {feat.shape}")
    
    # Forward pass
    outputs = decoder(features)
    
    print("\nOutput predictions:")
    for key, pred in outputs.items():
        print(f"  {key}: {pred.shape}")
    
    # Verify shapes
    assert outputs['main'].shape == (batch_size, 1, H, W), "Main output shape mismatch!"
    assert outputs['aux_1_8'].shape == (batch_size, 1, H//8, W//8), "Aux 1/8 shape mismatch!"
    assert outputs['aux_1_16'].shape == (batch_size, 1, H//16, W//16), "Aux 1/16 shape mismatch!"
    assert outputs['aux_1_32'].shape == (batch_size, 1, H//32, W//32), "Aux 1/32 shape mismatch!"
    
    print("\n✓ PPDDecoder unit test passed!")
