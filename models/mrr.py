"""
Mamba Reverse Refinement (MRR) Module

This module replaces the RA-RA (Residual Axial Reverse Attention) in ColonFormer
with a Mamba-based refinement using Cross-Scan SS2D.

Key advantages over RA-RA:
1. Truly global context (vs. axial decomposition that loses cross-spatial correlation)
2. Linear complexity O(N) (vs. quadratic O(N²) for attention)
3. Better boundary refinement through 4-directional cross-scanning

The "reverse" mechanism focuses attention on uncertain regions (boundaries and background)
rather than confident foreground regions, enabling precise polyp edge refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba_utils import SS2D


class MambaReverseRefinement(nn.Module):
    """
    Mamba-based Reverse Refinement Module for boundary refinement.
    
    Process flow:
    1. Reverse Masking: Create attention mask focusing on uncertain regions
       A = 1 - sigmoid(P_coarse), where P_coarse is the coarse prediction
    2. Guided Feature: Apply mask to features
       F' = F ⊙ A (element-wise multiplication)
    3. Cross-Scan Refinement: Process masked features with SS2D
       F_refined = SS2D(F')
    4. Residual Connection: Add refinement to original features
       Output = F + F_refined
    
    Args:
        dim (int): Feature dimension (number of channels)
        d_state (int): SSM state dimension (default: 16)
        d_conv (int): Local convolution width (default: 4)
        expand (int): Expansion factor (default: 2)
        num_heads (int): Number of refinement heads (default: 1)
    """
    
    def __init__(
        self, 
        dim, 
        d_state=16, 
        d_conv=4, 
        expand=2,
        num_heads=1,
        dropout=0.,
    ):
        super(MambaReverseRefinement, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Pre-normalization for features
        self.norm_feature = nn.LayerNorm(dim)
        
        # Prediction normalization (if prediction has multiple channels)
        self.norm_pred = nn.BatchNorm2d(1, eps=1e-3)  # Predictions are typically 1 channel
        
        # Feature projection before refinement (helps with feature conditioning)
        self.pre_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-3),
            nn.GELU()
        )
        
        # Core SS2D refinement module
        self.ss2d_refine = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # Post-refinement projection
        self.post_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-3),
            nn.GELU()
        )
        
        # Learnable gate to control residual strength
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, feature, prediction):
        """
        Forward pass through Mamba Reverse Refinement.
        
        Args:
            feature (Tensor): Feature map from decoder, shape [B, C, H, W]
            prediction (Tensor): Coarse prediction mask, shape [B, 1, H, W] (logits)
        
        Returns:
            Tensor: Refined features, shape [B, C, H, W]
        """
        B, C, H, W = feature.shape
        
        # Save original features for residual
        identity = feature
        
        # === Step 1: Reverse Masking ===
        # Normalize prediction
        pred_norm = self.norm_pred(prediction)
        
        # Create reverse attention mask (focus on uncertain/background regions)
        # Higher values where prediction is uncertain
        reverse_mask = 1.0 - torch.sigmoid(pred_norm)  # [B, 1, H, W]
        
        # Optional: Sharpen the mask (focus more on boundaries)
        # reverse_mask = torch.pow(reverse_mask, 0.5)  # Adjust exponent as needed
        
        # === Step 2: Guided Feature Extraction ===
        # Apply reverse mask to features (element-wise multiplication)
        # This makes the model focus on refining uncertain regions
        feature_masked = feature * reverse_mask  # [B, C, H, W]
        
        # === Step 3: Pre-projection ===
        feature_proj = self.pre_proj(feature_masked)  # [B, C, H, W]
        
        # === Step 4: Cross-Scan Refinement with SS2D ===
        # Convert to channel-last format for SS2D (VMamba uses BHWC)
        feature_proj_cl = feature_proj.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Normalize before SS2D
        feature_proj_cl = self.norm_feature(feature_proj_cl)
        
        # Apply SS2D with 4-directional cross-scanning
        feature_refined_cl = self.ss2d_refine(feature_proj_cl)  # [B, H, W, C]
        
        # Convert back to channel-first format
        feature_refined = feature_refined_cl.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # === Step 5: Post-projection ===
        feature_refined = self.post_proj(feature_refined)  # [B, C, H, W]
        
        # === Step 6: Gated Residual Connection ===
        # Use learnable gate to control how much refinement to add
        output = identity + self.gate * feature_refined
        
        return output
    
    def extra_repr(self):
        """Extra representation for printing"""
        return f'dim={self.dim}, num_heads={self.num_heads}, gate={self.gate.item():.3f}'


class MultiScaleMRR(nn.Module):
    """
    Multi-Scale Mamba Reverse Refinement.
    
    Applies MRR at multiple decoder scales as specified in the architecture:
    - Scale 2 (1/8 resolution)
    - Scale 3 (1/16 resolution)  
    - Scale 4 (1/32 resolution)
    
    Args:
        dims (list): List of feature dimensions at each scale [dim_2, dim_3, dim_4]
        d_state (int): SSM state dimension
        d_conv (int): Convolution width
        expand (int): Expansion factor
    """
    
    def __init__(self, dims=[128, 384, 768], d_state=16, d_conv=4, expand=2):
        super(MultiScaleMRR, self).__init__()
        
        assert len(dims) == 3, "Must provide dimensions for scales 2, 3, 4"
        
        # Create MRR module for each scale
        self.mrr_scale_2 = MambaReverseRefinement(
            dim=dims[0], d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mrr_scale_3 = MambaReverseRefinement(
            dim=dims[1], d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mrr_scale_4 = MambaReverseRefinement(
            dim=dims[2], d_state=d_state, d_conv=d_conv, expand=expand
        )
    
    def forward(self, features, predictions):
        """
        Apply MRR at multiple scales.
        
        Args:
            features (dict): Dictionary of features at each scale
                {'scale_2': [B, C2, H/8, W/8],
                 'scale_3': [B, C3, H/16, W/16],
                 'scale_4': [B, C4, H/32, W/32]}
            predictions (dict): Dictionary of predictions at each scale
                {'scale_2': [B, 1, H/8, W/8],
                 'scale_3': [B, 1, H/16, W/16],
                 'scale_4': [B, 1, H/32, W/32]}
        
        Returns:
            dict: Refined features at each scale with same keys
        """
        refined = {}
        
        if 'scale_2' in features:
            refined['scale_2'] = self.mrr_scale_2(
                features['scale_2'], predictions['scale_2']
            )
        
        if 'scale_3' in features:
            refined['scale_3'] = self.mrr_scale_3(
                features['scale_3'], predictions['scale_3']
            )
        
        if 'scale_4' in features:
            refined['scale_4'] = self.mrr_scale_4(
                features['scale_4'], predictions['scale_4']
            )
        
        return refined


if __name__ == '__main__':
    # Unit test
    print("Testing MambaReverseRefinement module...")
    
    # Test single scale
    batch_size = 2
    dim = 384
    H, W = 44, 44  # 1/8 scale of 352x352
    
    mrr = MambaReverseRefinement(dim=dim, d_state=16, d_conv=4, expand=2)
    print(f"MRR parameters: {sum(p.numel() for p in mrr.parameters()):,}")
    
    # Create dummy inputs
    feature = torch.randn(batch_size, dim, H, W)
    prediction = torch.randn(batch_size, 1, H, W)  # Coarse prediction logits
    
    print(f"Input feature shape: {feature.shape}")
    print(f"Input prediction shape: {prediction.shape}")
    
    # Forward pass
    output = mrr(feature, prediction)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == feature.shape, "Output shape mismatch!"
    print("✓ MRR single scale test passed!")
    
    # Test multi-scale
    print("\nTesting MultiScaleMRR...")
    multi_mrr = MultiScaleMRR(dims=[128, 384, 768])
    
    features = {
        'scale_2': torch.randn(batch_size, 128, 44, 44),   # 1/8
        'scale_3': torch.randn(batch_size, 384, 22, 22),   # 1/16
        'scale_4': torch.randn(batch_size, 768, 11, 11),   # 1/32
    }
    
    predictions = {
        'scale_2': torch.randn(batch_size, 1, 44, 44),
        'scale_3': torch.randn(batch_size, 1, 22, 22),
        'scale_4': torch.randn(batch_size, 1, 11, 11),
    }
    
    refined = multi_mrr(features, predictions)
    
    for scale, feat in refined.items():
        print(f"{scale}: {feat.shape}")
        assert feat.shape == features[scale].shape, f"{scale} shape mismatch!"
    
    print("✓ MultiScaleMRR test passed!")
