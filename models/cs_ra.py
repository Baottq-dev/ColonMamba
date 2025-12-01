"""
CS-RA: Cross-Scan Reverse Attention Module
Refinement module using SS2D cross-scan and reverse masking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba_utils import SS2D
class CSRA(nn.Module):
    """
    Cross-Scan Reverse Attention
    
    From diagram:
    1. Cross-Scan (SS2D) on features
    2. Reverse masking (1 - sigmoid)
    3. Attention mechanism
    4. Gated residual
    
    Args:
        dim: Feature dimension (channels)
        d_state: SSM state dimension
        d_conv: Conv kernel size for SS2D
        expand: Expansion factor
    """
    
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        self.dim = dim
        
        # Feature normalization
        self.norm = nn.BatchNorm2d(dim)
        
        # Cross-Scan module (SS2D from VMamba)
        self.cross_scan = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=0.0
        )
        
        # Mask normalization
        self.mask_norm = nn.BatchNorm2d(1)
        
        # Pre/post projections
        self.pre_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.post_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Learnable gate
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, feature, mask_pred):
        """
        Args:
            feature: [B, C, H, W] - Encoder feature (after CFP)
            mask_pred: [B, 1, H, W] - Previous prediction
        
        Returns:
            refined: [B, C, H, W] - Refined features
        """
        B, C, H, W = feature.shape
        
        # Normalize and project
        feat_norm = self.norm(feature)
        feat_proj = self.pre_proj(feat_norm)
        
        # Cross-Scan (SS2D requires BHWC format)
        feat_hwc = feat_proj.permute(0, 2, 3, 1).contiguous()
        cs_out = self.cross_scan(feat_hwc)  # [B, H, W, C]
        cs_out = cs_out.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # Reverse masking
        mask_norm = self.mask_norm(mask_pred)
        mask_sig = torch.sigmoid(mask_norm)
        reverse_mask = 1 - mask_sig  # Focus on uncertain regions
        
        # Expand mask to match channels
        mask_expanded = reverse_mask.expand(-1, C, -1, -1)
        
        # Apply attention
        attended = cs_out * mask_expanded
        
        # Post projection
        refined = self.post_proj(attended)
        
        # Gated residual
        output = feature + self.gate * refined
        
        return output