"""
ColonMamba - Matching ColonFormer Architecture
Changes from ColonFormer:
1. Encoder: MiT-B3 → Hybrid Res-VMamba
2. Refinement: aa_kernel → MRR (Mamba Cross-Scan)
Everything else IDENTICAL to ColonFormer!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hybrid_encoder import HybridResVMambaEncoder
from .uper_decoder import UPerHead
from .cfp_module import CFPModule
from .mrr import MambaReverseRefinement
from .conv_layer import Conv
class ColonMamba(nn.Module):
    """
    ColonMamba - Polyp Segmentation Model
    
    Architecture matches ColonFormer exactly except:
    - Encoder: Hybrid Res-VMamba instead of MiT-B3
    - Refinement: MRR instead of Axial Attention
    """
    
    def __init__(
        self,
        num_classes=1,
        pretrained_resnet=True,
        pretrained_vmamba=True,
        vmamba_checkpoint_path=None,
        channel_mode='project',  # 'project' or 'adapt'
        freeze_bn=False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.channel_mode = channel_mode
        
        # ============ Encoder (CHANGED from ColonFormer) ============
        self.encoder = HybridResVMambaEncoder(
            pretrained_resnet=pretrained_resnet,
            pretrained_vmamba=pretrained_vmamba,
            vmamba_checkpoint_path=vmamba_checkpoint_path,
            freeze_bn=freeze_bn,
            channel_mode=channel_mode,
        )
        
        # Get encoder channels (after projection if mode='project')
        encoder_channels = self.encoder.get_feature_channels()
        # project mode: [64, 128, 320, 512] - matches ColonFormer
        # adapt mode: [64, 128, 768, 768] - VMamba native
        
        c1, c2, c3, c4 = encoder_channels
        
        # ============ Decoder (SAME as ColonFormer) ============
        self.decode_head = UPerHead(
            in_channels=encoder_channels,
            channels=64,  # Unified decoder channels
            num_classes=num_classes,
        )
        
        # ============ CFP Modules (SAME as ColonFormer) ============
        # Applied on ENCODER features, NOT decoder output
        self.CFP_1 = CFPModule(c2, d=8)  # For x2
        self.CFP_2 = CFPModule(c3, d=8)  # For x3
        self.CFP_3 = CFPModule(c4, d=8)  # For x4
        
        # ============ MRR Modules (CHANGED: MRR instead of aa_kernel) ============
        # These replace AA_kernel in ColonFormer
        self.mrr_1 = MambaReverseRefinement(dim=c2, d_state=16, d_conv=4, expand=2)
        self.mrr_2 = MambaReverseRefinement(dim=c3, d_state=16, d_conv=4, expand=2)
        self.mrr_3 = MambaReverseRefinement(dim=c4, d_state=16, d_conv=4, expand=2)
        
        # ============ RA Convolutions (SAME as ColonFormer) ============
        # Refinement conv layers
        self.ra1_conv1 = Conv(c2, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra2_conv1 = Conv(c3, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra3_conv1 = Conv(c4, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
    
    def forward(self, x):
        """
        Forward pass - MATCHES ColonFormer flow exactly!
        
        Args:
            x: Input [B, 3, H, W]
            
        Returns:
            Tuple of (lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1)
        """
        # ============ Encoder ============
        features = self.encoder(x)
        x1, x2, x3, x4 = features
        
        # ============ UPerHead Decoder ============
        decoder_1 = self.decode_head([x1, x2, x3, x4])  # [B, 1, H/4, W/4]
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ============ Refinement Stage 1 (1/32 scale) ============
        decoder_2 = F.interpolate(decoder_1, size=x4.shape[2:], mode='bilinear')
        cfp_out_1 = self.CFP_3(x4)  # CFP on encoder x4
        
        # Reverse masking
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        
        # MRR (REPLACES aa_kernel)
        mrr_atten_3 = self.mrr_3(cfp_out_1, decoder_2)
        mrr_atten_3_o = decoder_2_ra.expand(-1, cfp_out_1.size(1), -1, -1).mul(mrr_atten_3)
        
        # RA convolutions
        ra_3 = self.ra3_conv1(mrr_atten_3_o)
        ra_3 = self.ra3_conv2(ra_3)
        ra_3 = self.ra3_conv3(ra_3)
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear')
        
        # ============ Refinement Stage 2 (1/16 scale) ============
        decoder_3 = F.interpolate(x_3, size=x3.shape[2:], mode='bilinear')
        cfp_out_2 = self.CFP_2(x3)  # CFP on encoder x3
        
        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        
        # MRR (REPLACES aa_kernel_2)
        mrr_atten_2 = self.mrr_2(cfp_out_2, decoder_3)
        mrr_atten_2_o = decoder_3_ra.expand(-1, cfp_out_2.size(1), -1, -1).mul(mrr_atten_2)
        
        ra_2 = self.ra2_conv1(mrr_atten_2_o)
        ra_2 = self.ra2_conv2(ra_2)
        ra_2 = self.ra2_conv3(ra_2)
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear')
        
        # ============ Refinement Stage 3 (1/8 scale) ============
        decoder_4 = F.interpolate(x_2, size=x2.shape[2:], mode='bilinear')
        cfp_out_3 = self.CFP_1(x2)  # CFP on encoder x2
        
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        
        # MRR (REPLACES aa_kernel_1)
        mrr_atten_1 = self.mrr_1(cfp_out_3, decoder_4)
        mrr_atten_1_o = decoder_4_ra.expand(-1, cfp_out_3.size(1), -1, -1).mul(mrr_atten_1)
        
        ra_1 = self.ra1_conv1(mrr_atten_1_o)
        ra_1 = self.ra1_conv2(ra_1)
        ra_1 = self.ra1_conv3(ra_1)
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear')
        
        # Return in SAME order as ColonFormer
        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1

    def get_model_size(self):
        """
        Calculate model parameters and size.
        
        Returns:
            tuple: (param_count, size_mb)
        """
        # Count total parameters
        param_count = sum(p.numel() for p in self.parameters())
        
        # Calculate size in MB (4 bytes per float32)
        size_mb = param_count * 4 / (1024 * 1024)
        
        return param_count, size_mb