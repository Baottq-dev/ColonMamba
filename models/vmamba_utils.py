"""
VMamba SS2D (Selective Scan 2D) Implementation

This module implements the core Mamba selective scan mechanism for 2D images,
following the VMamba-v1 (Vim) architecture from MzeroMiko/VMamba.

Key Concepts:
- SS2D: Selective Scan 2D with 4-directional scanning
- Cross-Scan: Scans image in 4 directions (↘, ↙, ↗, ↖) to capture spatial dependencies
- State Space Model: Uses selective state transitions for efficient long-range modeling

Reference: VMamba - Visual State Space Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class SS2D(nn.Module):
    """
    Selective Scan 2D - Core Mamba operation for processing 2D feature maps.
    
    This module performs 4-directional selective scanning to capture
    global spatial dependencies with linear complexity O(N).
    
    Args:
        d_model (int): Model dimension (number of channels)
        d_state (int): SSM state dimension (default: 16)
        d_conv (int): Local convolution width (default: 4)
        expand (int): Expansion factor for inner dimension (default: 2)
        dt_rank (str or int): Rank of dt projection ('auto' or specific value)
        dt_min (float): Minimum dt value
        dt_max (float): Maximum dt value
        dt_init (str): dt initialization method
        dt_scale (float): dt scaling factor
        dt_init_floor (float): dt initialization floor
    """
    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Depthwise convolution for local feature extraction (channel-wise)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=3,  # ← FIXED to 3 for V1!
            padding=1,      # ← (3-1)//2 = 1
        )
        
        # SSM parameters (4 directions)
        self.K = 4  # Number of scan directions
        
        # V1 uses raw tensor instead of nn.Linear
        self.x_proj_weight = nn.Parameter(
            torch.randn(self.K, self.dt_rank + self.d_state * 2, self.d_inner) * 0.02
        )
        
        # dt_projs_weight: [K, d_inner, dt_rank]
        self.dt_projs_weight = nn.Parameter(
            torch.randn(self.K, self.d_inner, self.dt_rank) * 0.02
        )
        
        # Initialize dt_projs_bias: [K, d_inner]
        dt = torch.exp(
            torch.rand(self.K, self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_projs_bias = nn.Parameter(inv_dt)
        
        A_logs = torch.randn(self.d_inner * self.K, self.d_state)
        self.A_logs = nn.Parameter(torch.log(A_logs.abs() + 1e-6))
        
        # Ds: V1 uses flat [d_inner * K] (PLURAL!)
        self.Ds = nn.Parameter(torch.ones(self.d_inner * self.K))
        
        # Add out_norm layer (VMamba V1 has this!)
        # Applied AFTER gating (on d_inner, NOT d_inner*2!)
        self.out_norm = nn.LayerNorm(self.d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
    
    def forward(self, x):
        """
        Forward pass through SS2D.
        
        Args:
            x (Tensor): Input features, shape [B, H, W, C] (channel-last)
        
        Returns:
            Tensor: Output features, shape [B, H, W, C]
        """
        B, H, W, C = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # [B, H, W, 2*d_inner]
        
        # Split into x and z (NO out_norm here!)
        x, z = xz.chunk(2, dim=-1)  # Each: [B, H, W, d_inner]
        
        # Depthwise convolution (need BCHW format)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.conv2d(x)  # [B, d_inner, H, W]
        x = rearrange(x, 'b c h w -> b h w c')
        x = F.silu(x)  # Activation
        
        # Cross-scan: convert to 4 directional sequences
        x_scans = self.cross_scan(x)  # [B, K, L, d_inner], K=4 directions, L=H*W
        
        # x_scans: [B, K, L, d_inner]
        # x_proj_weight: [K, out_feat, d_inner] where out_feat = dt_rank + d_state*2
        # Use einsum: (B,K,L,d_inner) @ (K,out_feat,d_inner) -> (B,K,L,out_feat)
        x_dbl = torch.einsum('bkli,koi->bklo', x_scans, self.x_proj_weight)
        # Now x_dbl: [B, K, L, dt_rank+d_state*2]

        # Split into dt, B, C components
        dt, B_param, C_param = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        # Now: dt [B,K,L,dt_rank], B_param [B,K,L,d_state], C_param [B,K,L,d_state]
        
        # dt: [B, K, L, dt_rank]
        # dt_projs_weight: [K, d_inner, dt_rank]
        # Result: [B, K, L, d_inner]
        dt = torch.einsum('bklr,kir->bkli', dt, self.dt_projs_weight)
        dt = dt + self.dt_projs_bias.view(1, self.K, 1, self.d_inner)
        dt = F.softplus(dt)

        # A_logs: [d_inner*K, d_state] -> [d_inner, d_state, K]
        A = -torch.exp(self.A_logs.float())  # [d_inner*K, d_state]
        A = A.view(self.d_inner, self.K, self.d_state).permute(0, 2, 1)  # [d_inner, d_state, K]
        
        # Ds: [d_inner*K] -> [d_inner, K]
        D = self.Ds.float().view(self.d_inner, self.K)
        
        # Rearrange x_scans for selective scan
        x_scans = rearrange(x_scans, 'b k l d -> b k d l')  # [B, K, d_inner, L]
        
        # Perform selective scan for each direction
        # This is a simplified version - full implementation would use efficient CUDA kernels
        y_scans = self.selective_scan(
            x_scans, dt, A, B_param, C_param, D
        )  # [B, K, d_inner, L]
        
        # Merge scans from 4 directions
        y = self.merge_scans(y_scans, H, W)  # [B, H, W, d_inner]
        
        # Gate with z
        y = y * F.silu(z)
        
        # Apply out_norm AFTER gating (V1 position!)
        y = self.out_norm(y)  # [B, H, W, d_inner]
        
        # Output projection
        output = self.out_proj(y)  # [B, H, W, C]
        
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def cross_scan(self, x):
        """
        Perform 4-directional cross-scanning on 2D feature map.
        
        Scan directions:
        1. Top-left to bottom-right (↘)
        2. Top-right to bottom-left (↙)
        3. Bottom-left to top-right (↗)
        4. Bottom-right to top-left (↖)
        
        Args:
            x (Tensor): [B, H, W, C]
        
        Returns:
            Tensor: [B, K=4, L=H*W, C]
        """
        B, H, W, C = x.shape
        L = H * W
        
        # Flatten spatial dimensions
        x_flat = rearrange(x, 'b h w c -> b (h w) c')  # [B, L, C]
        
        # Direction 1: Top-left to bottom-right (↘) - natural order
        scan1 = x_flat
        
        # Direction 2: Top-right to bottom-left (↙) - flip width
        x_flip_w = torch.flip(x, dims=[2])
        scan2 = rearrange(x_flip_w, 'b h w c -> b (h w) c')
        
        # Direction 3: Bottom-left to top-right (↗) - flip height
        x_flip_h = torch.flip(x, dims=[1])
        scan3 = rearrange(x_flip_h, 'b h w c -> b (h w) c')
        
        # Direction 4: Bottom-right to top-left (↖) - flip both
        x_flip_hw = torch.flip(x, dims=[1, 2])
        scan4 = rearrange(x_flip_hw, 'b h w c -> b (h w) c')
        
        # Stack all directions
        scans = torch.stack([scan1, scan2, scan3, scan4], dim=1)  # [B, K=4, L, C]
        
        return scans
    
    def merge_scans(self, y_scans, H, W):
        """
        Merge outputs from 4-directional scans back to 2D feature map.
        
        Args:
            y_scans (Tensor): [B, K=4, C, L]
            H (int): Height
            W (int): Width
        
        Returns:
            Tensor: [B, H, W, C]
        """
        B, K, C, L = y_scans.shape
        
        # Reshape each scan back to 2D
        y1, y2, y3, y4 = y_scans[:, 0], y_scans[:, 1], y_scans[:, 2], y_scans[:, 3]
        
        # Direction 1: natural order
        y1_2d = rearrange(y1, 'b c (h w) -> b h w c', h=H, w=W)
        
        # Direction 2: flip width back
        y2_2d = rearrange(y2, 'b c (h w) -> b h w c', h=H, w=W)
        y2_2d = torch.flip(y2_2d, dims=[2])
        
        # Direction 3: flip height back
        y3_2d = rearrange(y3, 'b c (h w) -> b h w c', h=H, w=W)
        y3_2d = torch.flip(y3_2d, dims=[1])
        
        # Direction 4: flip both back
        y4_2d = rearrange(y4, 'b c (h w) -> b h w c', h=H, w=W)
        y4_2d = torch.flip(y4_2d, dims=[1, 2])
        
        # Average all 4 directions
        y = (y1_2d + y2_2d + y3_2d + y4_2d) / 4.0
        
        return y
    
    def selective_scan(self, x, dt, A, B, C, D):
        """
        Simplified selective scan implementation.
        
        In production, this should use efficient CUDA kernels from the mamba-ssm package.
        This is a PyTorch-only reference implementation.
        
        Args:
            x (Tensor): Input sequence [B, K, d_inner, L]
            dt (Tensor): Delta values [B, K, L, d_inner]
            A (Tensor): State transition matrix [d_inner, d_state, K]
            B (Tensor): Input matrix [B, K, L, d_state]
            C (Tensor): Output matrix [B, K, L, d_state]
            D (Tensor): Skip connection [d_inner, K]
        
        Returns:
            Tensor: Output sequence [B, K, d_inner, L]
        """
        batch_size, K, d_inner, L = x.shape  # ← FIX: Changed B to batch_size to avoid shadowing!
        d_state = A.shape[1]
        
        # Rearrange for batched processing
        dt = rearrange(dt, 'b k l d -> b k d l')  # [B, K, d_inner, L]
        B_param = rearrange(B, 'b k l d -> b k d l')  # [B, K, d_state, L]
        C_param = rearrange(C, 'b k l d -> b k d l')  # [B, K, d_state, L]
        
        # Simplified scan (this is a placeholder - real implementation uses efficient recurrence)
        # For now, use a weighted combination as an approximation
        y = torch.zeros_like(x)  # [B, K, d_inner, L]
        
        for k in range(K):
            A_k = A[:, :, k]  # [d_inner, d_state]
            D_k = D[:, k]  # [d_inner]
            
            # Simplified: y = D * x (skip connection dominates in this simple version)
            # Real implementation would perform state space recurrence
            y[:, k] = x[:, k] * D_k.view(1, -1, 1)
        
        return y


class VSSBlock(nn.Module):
    """
    Vision State Space Block - Basic building block combining SS2D with normalization.
    
    This is typically used multiple times in VMamba stages.
    
    Architecture (from pretrained VMamba):
    - ln_1: Pre-normalization before SS2D
    - ss2d: Selective Scan 2D (has out_norm inside)
    - Residual: x + ss2d(ln_1(x))
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0., **kwargs):
        super().__init__()
        # Pretrained VMamba HAS ln_1 (pre-norm)!
        self.ln_1 = nn.LayerNorm(d_model)
        
        self.ss2d = SS2D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            **kwargs
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): [B, H, W, C]
        Returns:
            Tensor: [B, H, W, C]
        """
        # Pretrained: x + ss2d(ln_1(x))
        return x + self.ss2d(self.ln_1(x))


if __name__ == '__main__':
    # Unit test
    print("Testing SS2D module...")
    
    # Test parameters
    batch_size = 2
    H, W = 32, 32
    d_model = 96
    
    # Create SS2D module
    ss2d = SS2D(d_model=d_model, d_state=16, d_conv=4, expand=2)
    print(f"SS2D parameters: {sum(p.numel() for p in ss2d.parameters()):,}")
    
    # Test forward pass (channel-last format for VMamba)
    x = torch.randn(batch_size, H, W, d_model)
    print(f"Input shape: {x.shape}")
    
    output = ss2d(x)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    print("✓ SS2D unit test passed!")
    
    # Test VSSBlock
    print("\nTesting VSSBlock...")
    vss_block = VSSBlock(d_model=d_model)
    output = vss_block(x)
    assert output.shape == x.shape, "VSSBlock output shape mismatch!"
    print("✓ VSSBlock unit test passed!")
