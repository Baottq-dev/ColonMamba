"""
Local-Global Refinement Block for ColonFormer
2-Branch Bottleneck: Residual + (Local DW-Conv || Global SS2D/AA_kernel)

Created for ColonFormer improvement research.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.backbones.vmamba import SS2D
from .axial_atten import AA_kernel


class LocalGlobalBlock(nn.Module):
    """
    2-Branch Bottleneck Block for feature refinement.
    
    Architecture:
        Input (C) ─────────────────────────────────────┐
            │                                          │ (Residual)
            ▼                                          │
        Compress: Conv 1×1 (C → C/r)                   │
            │                                          │
        ┌───┴───┐                                      │
        │       │                                      │
        ▼       ▼                                      │
     DW-Conv  Global                                   │
     (Local)  (SS2D/AA)                                │
        │       │                                      │
        └───┬───┘                                      │
            │ (+) Fusion                               │
            ▼                                          │
        Expand: Conv 1×1 (C/r → C)                     │
            │                                          │
            └──────────────┬───────────────────────────┘
                           │ (+) Residual
                           ▼
                        Output (C)
    
    Args:
        channels: Number of input/output channels
        reduction: Bottleneck reduction ratio (default: 2)
        global_module: Pre-built global attention module (SS2D or AA_kernel)
    """
    
    def __init__(self, channels, reduction=2, global_module=None):
        super().__init__()
        
        self.channels = channels
        self.hidden = channels // reduction
        
        # ========== Bottleneck Entrance ==========
        # Conv 1×1 to reduce channels and mix features
        self.compress = nn.Sequential(
            nn.Conv2d(channels, self.hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True)
        )
        
        # ========== Local Branch: Depthwise Conv 3×3 ==========
        # Captures local features (edges, textures) without channel mixing
        self.local_branch = nn.Sequential(
            nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1, 
                      groups=self.hidden, bias=False),  # Depthwise separable
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(inplace=True)
        )
        
        # ========== Global Branch: SS2D or AA_kernel ==========
        # Captures global context and long-range dependencies
        if global_module is not None:
            self.global_branch = global_module
        else:
            # Fallback: identity (should not happen in normal use)
            self.global_branch = nn.Identity()
        
        # ========== Bottleneck Exit ==========
        # Conv 1×1 to expand channels back and fuse features
        self.expand = nn.Sequential(
            nn.Conv2d(self.hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # ========== Learnable Residual Weight ==========
        # Initialized to small value for stable training start
        # As training progresses, model learns optimal weight
        self.gamma = nn.Parameter(torch.ones(1) * 0.05)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W) - NCHW format
        
        Returns:
            Output tensor (B, C, H, W) - same shape as input
        """
        # ========== Compress: C → C/r ==========
        out = self.compress(x)
        
        # ========== Parallel Branches in Bottleneck Space ==========
        local_out = self.local_branch(out)     # Local features
        global_out = self.global_branch(out)   # Global context
        
        # ========== Fusion: Element-wise Addition ==========
        # Local and global features are combined in reduced space
        out = local_out + global_out
        
        # ========== Expand: C/r → C ==========
        out = self.expand(out)
        
        # ========== Return processed output (no residual) ==========
        # gamma scales the contribution, residual is added in colonformer.py
        return self.gamma * out


def build_ss2d_module(channels, d_state=1, ssm_ratio=2.0, d_conv=3, 
                      forward_type='v05_noz', channel_first=True):
    """
    Build SS2D module for global branch.
    
    Args:
        channels: Number of input channels
        d_state: SSM state dimension (default: 1 for efficiency)
        ssm_ratio: SSM expansion ratio (default: 2.0)
        d_conv: Depthwise conv kernel size (default: 3)
        forward_type: SS2D forward type (default: 'v05_noz')
        channel_first: Whether input is NCHW (default: True)
    
    Returns:
        SS2D module instance
    """
    
    return SS2D(
        d_model=channels,
        d_state=d_state,
        ssm_ratio=ssm_ratio,
        dt_rank='auto',
        d_conv=d_conv,
        forward_type=forward_type,
        channel_first=channel_first
    )


def build_aa_kernel_module(channels):
    """
    Build AA_kernel module for global branch.
    
    Args:
        channels: Number of input channels
    
    Returns:
        AA_kernel module instance
    """
        
    return AA_kernel(channels, channels)


def build_local_global_block(channels, attention_type='ss2d', reduction=2):
    """
    Factory function to build LocalGlobalBlock.
    
    This is the recommended way to create LocalGlobalBlock instances,
    as it handles the creation of the appropriate global module.
    
    Args:
        channels: Number of channels
        attention_type: Type of global attention ('ss2d' or 'aa_kernel')
        reduction: Bottleneck reduction ratio (default: 2)
    
    Returns:
        LocalGlobalBlock instance
    
    Example:
        >>> block = build_local_global_block(192, 'ss2d', reduction=2)
        >>> x = torch.randn(2, 192, 44, 44)
        >>> out = block(x)
        >>> print(out.shape)  # (2, 192, 44, 44)
    """
    hidden = channels // reduction
    
    if attention_type == 'ss2d':
        global_module = build_ss2d_module(
            channels=hidden,
            d_state=8,
            ssm_ratio=2.0,
            d_conv=3,
            forward_type='v05_noz',
            channel_first=True
        )
    elif attention_type == 'aa_kernel':
        global_module = build_aa_kernel_module(hidden)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}. "
                         f"Supported: 'ss2d', 'aa_kernel'")
    
    return LocalGlobalBlock(channels, reduction, global_module)


# ============================================================
# Unit Test (run with: python -m mmseg.models.segmentors.lib.local_global_block)
# ============================================================
if __name__ == '__main__':
    print("Testing LocalGlobalBlock...")
    
    # Test 1: Basic structure
    print("\n[Test 1] Creating block with mock global module...")
    mock_global = nn.Identity()
    block = LocalGlobalBlock(192, reduction=2, global_module=mock_global)
    x = torch.randn(2, 192, 44, 44)
    out = block(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"  Input: {x.shape} → Output: {out.shape} ✓")
    
    # Test 2: Gradient flow
    print("\n[Test 2] Checking gradient flow...")
    out.sum().backward()
    assert block.gamma.grad is not None, "Gamma has no gradient!"
    print(f"  Gamma gradient: {block.gamma.grad.item():.6f} ✓")
    
    # Test 3: Parameter count
    print("\n[Test 3] Parameter count...")
    params = sum(p.numel() for p in block.parameters())
    print(f"  Total parameters: {params:,}")
    
    print("\n✅ All tests passed!")
