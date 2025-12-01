"""
Hybrid ResNet-VMamba Encoder

Combines ResNet-34 (Stages 1-2) with VMamba-Tiny (Stages 3-4) for polyp segmentation.

Architecture:
- Stem: ResNet Conv7x7 + MaxPool → H/4 × W/4 (64 channels)  
- Stage 1: ResNet Layer1 → H/4 × W/4 (64 channels) [F1]
- Stage 2: ResNet Layer2 → H/8 × W/8 (128 channels) [F2]
- FTM Bridge: Conv+BN+GELU → H/8 × W/8 (384 channels) [F2']
- Stage 3: VMamba Layer3 → H/16 × W/16 (384 channels) [F3]
- Stage 4: VMamba Layer4 → H/32 × W/32 (768 channels) [F4]

Total parameters: ~40-50M (ResNet-34: ~21M + VMamba-Tiny: ~22M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import os
from .ftm_bridge import FTM_Bridge
from .vmamba_utils import VSSBlock
from einops import rearrange

class PatchMerging(nn.Module):
    """
    Patch Merging Layer for downsampling in VMamba stages.
    Reduces spatial dimensions by 2x and increases channels.
    """
    
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W, C]
        Returns:
            [B, H/2, W/2, 2*C]
        """
        B, H, W, C = x.shape
        
        # Ensure H and W are even
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            B, H, W, C = x.shape
        
        # Split into 4 patches and concatenate
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4*C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2, W/2, 2*C]
        
        return x


class VMambaStage(nn.Module):
    """
    VMamba Stage containing PatchMerging + multiple VSSBlocks.
    """
    
    def __init__(self, dim, out_dim, depth, d_state=16, d_conv=4, expand=2, downsample=True):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        
        # Downsampling layer
        if downsample:
            self.downsample = PatchMerging(dim, out_dim)
            block_dim = out_dim
        else:
            self.downsample = None
            block_dim = dim
        
        # Stack of VSS blocks
        self.blocks = nn.ModuleList([
            VSSBlock(
                d_model=block_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(depth)
        ])
        
        # Output norm
        self.norm = nn.LayerNorm(block_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W, C]
        Returns:
            [B, H', W', C'], where H'=H/2, W'=W/2 if downsample=True
        """
        # Downsample if needed
        if self.downsample is not None:
            x = self.downsample(x)
        
        # Pass through VSS blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class HybridResVMambaEncoder(nn.Module):
    """
    Hybrid Encoder combining ResNet-34 (early stages) and VMamba-Tiny (deep stages).
    
    Args:
        pretrained_resnet (bool): Use ImageNet pretrained ResNet (default: True)
        pretrained_vmamba (bool): Use pretrained VMamba weights (default: True)
        vmamba_checkpoint_path (str): Path to VMamba checkpoint file
        in_channels (int): Input image channels (default: 3)
        vmamba_depths (list): Number of blocks in each VMamba stage
    """
    
    def __init__(
        self,
        pretrained_resnet=True,
        pretrained_vmamba=True,
        vmamba_checkpoint_path=None,  
        in_channels=3,
        vmamba_depths=[9, 2],  # Stage 3: 9 blocks, Stage 4: 2 blocks (Vanilla VMamba-Tiny v0)
        freeze_bn=False,
        channel_mode='project',
    ):
        super(HybridResVMambaEncoder, self).__init__()

        self.in_channels = in_channels
        self.vmamba_checkpoint_path = vmamba_checkpoint_path
        self.channel_mode = channel_mode  # NEW: Store mode

        
        # ============ ENFORCE PRETRAINED REQUIREMENT ============
        if pretrained_vmamba:
            if vmamba_checkpoint_path is None:
                print("⚠️  Warning: pretrained_vmamba=True but path is None!")
                print("    VMamba will use RANDOM initialization (for testing only)")
                pretrained_vmamba = False
            elif not os.path.exists(vmamba_checkpoint_path):
                raise FileNotFoundError(
                    f"VMamba checkpoint not found at: {vmamba_checkpoint_path}\n"
                    f"Please download vssmtiny_dp01_ckpt_epoch_292.pth\n"
                    f"Download: https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth"
                )
        
        # ============ ResNet Stages (CNN Branch) ============
        # Load pretrained ResNet-34
        resnet = models.resnet34(pretrained=pretrained_resnet)
        
        # Stem: Conv7x7 + BN + ReLU + MaxPool → 1/4 resolution
        self.stem = nn.Sequential(
            resnet.conv1,      # Conv2d(3, 64, 7, stride=2, padding=3)
            resnet.bn1,        # BatchNorm2d(64)
            resnet.relu,       # ReLU
            resnet.maxpool,    # MaxPool2d(3, stride=2, padding=1)
        )
        
        # Stage 1: ResNet Layer1 → 1/4 resolution, 64 channels
        self.stage1 = resnet.layer1  # 3 BasicBlocks
        
        # Stage 2: ResNet Layer2 → 1/8 resolution, 128 channels  
        self.stage2 = resnet.layer2  # 4 BasicBlocks
        
        # Optional: Freeze BatchNorm layers in ResNet (helps with small batch sizes)
        if freeze_bn:
            self._freeze_bn()
        
        # ============ Feature Transition Module ============
        # Bridge from ResNet (128ch) to VMamba (384ch)
        self.ftm_bridge = FTM_Bridge(in_channels=128, out_channels=384)
        
        # ============ VMamba Stages (Global Context Branch) ============
        # Stage 3: VMamba Layer2 → 1/16 resolution, 384 channels
        # Maps to pretrained layers.2 (384→384, NO channel downsample)
        # Pretrained has patch merging AFTER blocks: 384→768
        self.stage3 = VMambaStage(
            dim=384,
            out_dim=384,  # Process at 384, let pretrained patch merging handle 384→768
            depth=vmamba_depths[0],
            downsample=False,  # No built-in downsample, use pretrained patch merging
        )
        
        # ============ Channel Projection 3→4 (Adapt Pretrained Weights) ============
        # Pretrained has: norm [1536] + reduction [768, 1536]
        # We can't use norm (wrong size), only use reduction weights!
        # Pretrained reduction expects [B, H/16, W/16, 1536] (after merging 4 patches of 384)
        # We have [B, H/16, W/16, 384] (no merging)
        # Solution: Use first 384 dims of reduction weight [768, 1536] → [768, 384]
        self.stage3_channel_proj = nn.Linear(384, 768, bias=False)

        # Stage 4: VMamba Layer3 → 1/16 resolution (NO additional spatial downsample!), 768 channels
        # Maps to pretrained layers.3 (768→768)
        # Note: Spatial is already at H/16, no more downsampling needed!
        self.stage4 = VMambaStage(
            dim=768,      # Input is 768 from channel projection!
            out_dim=768,
            depth=vmamba_depths[1],
            downsample=False,  # NO spatial downsample! Keep at 1/16
        )

        # ============ Channel Projection (for fair comparison) ============
        # NEW: Add projection layers based on mode
        if channel_mode == 'project':
            # Project to ColonFormer channels [320, 512]
            # Note: f3 is 384 (from stage3), f4 is 768 (after patch merging)
            self.channel_proj = nn.ModuleDict({
                'f3': nn.Sequential(
                    nn.Conv2d(384, 320, 1, bias=False),  # Stage 3 outputs 384 before patch merge
                    nn.BatchNorm2d(320),
                ),
                'f4': nn.Sequential(
                    nn.Conv2d(768, 512, 1, bias=False),  # Stage 4 outputs 768
                    nn.BatchNorm2d(512),
                ),
            })
        else:  # channel_mode == 'adapt'
            # No projection - use Identity (pass-through)
            self.channel_proj = nn.ModuleDict({
                'f3': nn.Identity(),
                'f4': nn.Identity(),
            })
        
        # ============ Load Pretrained Weights ============
        if pretrained_vmamba and vmamba_checkpoint_path is not None:
            self._load_vmamba_pretrained(vmamba_checkpoint_path)
    
    def _freeze_bn(self):
        """Freeze BatchNorm statistics in ResNet stages"""
        for module in [self.stem, self.stage1, self.stage2]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    
    def _load_vmamba_pretrained(self, checkpoint_path):
        """
        Load pretrained VMamba weights for Stage 3 and Stage 4.
        
        Filters out Stage 0 and Stage 1 weights (replaced by ResNet).
        Only loads 'layers.2.*' (Stage 3) and 'layers.3.*' (Stage 4).
        """
        print(f"Loading VMamba pretrained weights from: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter keys: only keep layers.2.* (Stage 3) and layers.3.* (Stage 4)
            stage3_dict = OrderedDict()
            stage4_dict = OrderedDict()
            patch_merge_dict = OrderedDict()  # For layers.2.downsample
            
            for key, value in state_dict.items():
                if 'layers.2.downsample' in key:
                    # Patch merging weights (layers.2.downsample)
                    new_key = key.replace('layers.2.downsample.', '')
                    patch_merge_dict[new_key] = value
                elif 'layers.2.' in key:
                    # Remove 'layers.2.' prefix and map to stage3
                    new_key = key.replace('layers.2.', '')
                    new_key = new_key.replace('ln_1', 'norm')
                    new_key = new_key.replace('self_attention', 'ss2d')
                    stage3_dict[new_key] = value
                elif 'layers.3.' in key:
                    # Remove 'layers.3.' prefix and map to stage4
                    new_key = key.replace('layers.3.', '')
                    new_key = new_key.replace('ln_1', 'norm')
                    new_key = new_key.replace('self_attention', 'ss2d')
                    stage4_dict[new_key] = value
            
            # Load filtered weights
            # 1. Load and adapt channel projection weights
            if len(patch_merge_dict) > 0:
                # Pretrained has reduction.weight [768, 1536] (expects 4 merged patches)
                # We need [768, 384] (only 1 patch, no merging)
                # Solution: Take first 384 columns of reduction weight
                if 'reduction.weight' in patch_merge_dict:
                    pretrained_reduction = patch_merge_dict['reduction.weight']  # [768, 1536]
                    # Take first 384 dims (corresponding to first patch in pretrained)
                    adapted_weight = pretrained_reduction[:, :384]  # [768, 384]
                    self.stage3_channel_proj.weight.data.copy_(adapted_weight)
                    print(f"Loaded & adapted channel projection weights: [768, 1536] -> [768, 384]")
                else:
                    print("Warning: No reduction.weight found in pretrained")
            
            # 2. Load Stage 3 weights
            if len(stage3_dict) > 0:
                missing, unexpected = self.stage3.load_state_dict(stage3_dict, strict=False)
                print(f"Loaded Stage 3 weights: {len(stage3_dict)} keys")
                if missing:
                    print(f"  Missing keys: {len(missing)}")
                    print(f"  First 10 missing keys:")
                    for i, key in enumerate(list(missing)[:10]):
                        print(f"    {i+1}. {key}")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)}")
            
            # 3. Load Stage 4 weights
            if len(stage4_dict) > 0:
                missing, unexpected = self.stage4.load_state_dict(stage4_dict, strict=False)
                print(f"Loaded Stage 4 weights: {len(stage4_dict)} keys")
                if missing:
                    print(f"  Missing keys: {len(missing)}")
                    print(f"  First 10 missing keys:")
                    for i, key in enumerate(list(missing)[:10]):
                        print(f"    {i+1}. {key}")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)}")
            
            if len(stage3_dict) == 0 and len(stage4_dict) == 0:
                print("Warning: No matching VMamba weights found in checkpoint!")
                
        except Exception as e:
            print(f"Error loading VMamba weights: {e}")
            print("VMamba stages will use random initialization.")
    
    def forward(self, x):
        """
        Forward pass through hybrid encoder.
        
        Args:
            x (Tensor): Input image, shape [B, 3, H, W]
        
        Returns:
            list: Multi-scale features [F1, F2, F3, F4]
                F1: [B, 64, H/4, W/4]
                F2: [B, 128, H/8, W/8]
                F3: [B, 384, H/16, W/16]
                F4: [B, 768, H/32, W/32]
        """
        # ============ CNN Branch (ResNet) ============
        # Stem → 1/4
        x = self.stem(x)  # [B, 64, H/4, W/4]
        
        # Stage 1 → 1/4
        F1 = self.stage1(x)  # [B, 64, H/4, W/4]
        
        # Stage 2 → 1/8
        F2 = self.stage2(F1)  # [B, 128, H/8, W/8]
        
        # ============ Feature Transition ============
        # Bridge: 128 → 384 channels (maintains 1/8 resolution)
        F2_proj = self.ftm_bridge(F2)  # [B, 384, H/8, W/8]
        
        # ============ VMamba Branch (Global Context) ============
        # Convert to channel-last format for VMamba (BHWC)
        F2_cl = F2_proj.permute(0, 2, 3, 1)  # [B, H/8, W/8, 384]
        
        # Stage 3 → 1/16
        F3_cl = self.stage3(F2_cl)  # [B, H/16, W/16, 384]
        
        # Apply channel projection: 384 → 768
        F3_proj_cl = self.stage3_channel_proj(F3_cl)  # [B, H/16, W/16, 768]
        
        # Stage 4 → still at 1/16 (pretrained has no spatial downsample!)
        F4_cl = self.stage4(F3_proj_cl)  # [B, H/16, W/16, 768]
        
        # Convert back to channel-first format (BCHW) for decoder
        F3 = F3_cl.permute(0, 3, 1, 2)  # [B, 384, H/16, W/16]
        F4 = F4_cl.permute(0, 3, 1, 2)  # [B, 768, H/16, W/16]
        
        # ============ MODE-DEPENDENT SPATIAL ADAPTATION ============
        # 'project' mode: Downsample F4 to H/32 to match ColonFormer decoder
        # 'adapt' mode: Keep F4 at H/16 (native VMamba output)
        if self.channel_mode == 'project':
            # Spatial downsample: H/16 → H/32
            F4 = F.avg_pool2d(F4, kernel_size=2, stride=2)  # [B, 768, H/32, W/32]
        # else: F4 stays at H/16 for 'adapt' mode
        
        # NEW: Apply channel projection
        F3 = self.channel_proj['f3'](F3)  # [B, 320 or 384, ...]
        F4 = self.channel_proj['f4'](F4)  # [B, 512, H/32, W/32] for 'project'
                                          # [B, 768, H/16, W/16] for 'adapt'

        # Return multi-scale features (all in BCHW format)
        return [F1, F2, F3, F4]
    
    def get_feature_channels(self):
        """
        Get output channel dimensions for each stage.
        
        Returns based on mode:
        - 'project': [64, 128, 320, 512] (matches ColonFormer)
        - 'adapt': [64, 128, 384, 768] (VMamba native: stage3=384, stage4=768)
        """
        if self.channel_mode == 'project':
            # For fair comparison with ColonFormer
            return [64, 128, 320, 512]
        else:
            # VMamba native dimensions
            return [64, 128, 384, 768]  # stage3=384, stage4=768

if __name__ == '__main__':
    # Unit test
    print("Testing HybridResVMambaEncoder...")
    
    # Create encoder (without pretrained VMamba for testing)
    encoder = HybridResVMambaEncoder(
        pretrained_resnet=False,  # Set to False for faster testing
        pretrained_vmamba=False,
    )
    
    print(f"Encoder: {encoder.get_feature_channels()}")
    print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 352, 352)
    print(f"Input shape: {x.shape}")
    
    features = encoder(x)
    
    print("\nOutput features:")
    for i, feat in enumerate(features, 1):
        print(f"  F{i}: {feat.shape} ({feat.shape[1]} channels)")
    
    # Verify shapes
    expected_shapes = [
        (batch_size, 64, 88, 88),    # 352/4 = 88
        (batch_size, 128, 44, 44),   # 352/8 = 44
        (batch_size, 384, 22, 22),   # 352/16 = 22
        (batch_size, 768, 11, 11),   # 352/32 = 11
    ]
    
    for i, (feat, expected) in enumerate(zip(features, expected_shapes), 1):
        assert feat.shape == expected, f"F{i} shape mismatch! Got {feat.shape}, expected {expected}"
    
    print("\n✓ HybridResVMambaEncoder unit test passed!")
