import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .. import builder

import numpy as np
import cv2

from .lib.conv_layer import Conv, BNPReLU
from .lib.axial_atten import AA_kernel
from .lib.context_module import CFPModule
from mmengine.runner import load_checkpoint
from ..utils.weight_converter import load_pretrained_mit


class SS2D_Wrapper(nn.Module):

    def __init__(self, d_model, ss2d_module):
        super().__init__()
        # Preprocessing convs
        self.conv0 = Conv(d_model, d_model, kSize=1, stride=1, padding=0, bn_acti=True)
        self.conv1 = Conv(d_model, d_model, kSize=3, stride=1, padding=1, bn_acti=True)
        
        # SS2D for spatial attention
        self.ss2d = ss2d_module
        
        # Learnable gamma, initialized to small value (NOT 0!)
        # CRITICAL: gamma=0 blocks gradient flow to SS2D (d/d(ss2d) = 0)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        # Preprocessing convs
        out = self.conv0(x)
        out = self.conv1(out)
        
        # SS2D attention with residual + gamma
        ss2d_out = self.ss2d(out)
        return self.gamma * ss2d_out + out


@MODELS.register_module()
class ColonFormer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_ss2d=False):
        super(ColonFormer, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        # Detect backbone type and get output channels
        backbone_type = backbone.get('type', 'MixVisionTransformer')
        if 'VSSM' in backbone_type or 'VMamba' in backbone_type:
            # VMamba: calculate channels from base dimension
            base_dim = backbone.get('dims', 96)
            if isinstance(base_dim, int):
                # Calculate channel progression: dims * 2^stage
                self.c1 = base_dim
                self.c2 = base_dim * 2
                self.c3 = base_dim * 4
                self.c4 = base_dim * 8
            else:
                # If dims is already a list [c1, c2, c3, c4]
                self.c1, self.c2, self.c3, self.c4 = base_dim
            print(f"[ColonFormer] Using VMamba backbone with channels: [{self.c1}, {self.c2}, {self.c3}, {self.c4}]")
        else:
            # SegFormer (MiT) channels: [64, 128, 320, 512]
            self.c1, self.c2, self.c3, self.c4 = 64, 128, 320, 512
            print(f"[ColonFormer] Using SegFormer backbone with channels: [{self.c1}, {self.c2}, {self.c3}, {self.c4}]")
        
        # Dynamic decoder initialization based on backbone channels
        self.CFP_1 = CFPModule(self.c2, d=8)
        self.CFP_2 = CFPModule(self.c3, d=8)
        self.CFP_3 = CFPModule(self.c4, d=8)

        self.ra1_conv1 = Conv(self.c2, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra2_conv1 = Conv(self.c3, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra3_conv1 = Conv(self.c4, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        # Spatial Attention: Hybrid mode (AA_kernel or SS2D)
        if use_ss2d:
            # Use SS2D from VMamba for better spatial modeling
            try:
                from mmseg.models.backbones.vmamba import SS2D
            except ImportError as e:
                raise ImportError(
                    "SS2D mode requires mamba-ssm to be installed. "
                    "Please run: pip install mamba-ssm (requires CUDA toolkit)\n"
                    f"Original error: {e}"
                )
            print("[ColonFormer] Using SS2D for spatial attention (VMamba-style)")
            # Wrap SS2D with conv preprocessing + residual to match AA_kernel
            self.aa_kernel_1 = SS2D_Wrapper(self.c2, SS2D(
                d_model=self.c2,
                d_state=1,           
                ssm_ratio=2.0,
                dt_rank='auto',
                d_conv=3,
                forward_type='v05_noz',  
                channel_first=True
            ))
            self.aa_kernel_2 = SS2D_Wrapper(self.c3, SS2D(
                d_model=self.c3,
                d_state=1,
                ssm_ratio=2.0,
                dt_rank='auto',
                d_conv=3,
                forward_type='v05_noz',
                channel_first=True
            ))
            self.aa_kernel_3 = SS2D_Wrapper(self.c4, SS2D(
                d_model=self.c4,
                d_state=1,
                ssm_ratio=2.0,
                dt_rank='auto',
                d_conv=3,
                forward_type='v05_noz',
                channel_first=True
            ))
        else:
            # Use original Axial Attention
            print("[ColonFormer] Using Axial Attention for spatial attention (original)")
            self.aa_kernel_1 = AA_kernel(self.c2, self.c2)
            self.aa_kernel_2 = AA_kernel(self.c3, self.c3)
            self.aa_kernel_3 = AA_kernel(self.c4, self.c4)
        
        # Initialize weights AFTER all modules are created
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.
        Args:
            pretrained (str): Path to pretrained backbone weights. Required.
        
        Raises:
            ValueError: If pretrained is None.
        """
        if pretrained is None:
            raise ValueError(
                "Please provide pretrained path when initializing the model."
            )
        
        # Check backbone type to decide loading method
        backbone_type = self.backbone.__class__.__name__
        if backbone_type == 'MixVisionTransformer':
            # Use weight converter for MiT backbone (handles old->new key format)
            load_pretrained_mit(self.backbone, pretrained)
        elif backbone_type == 'Backbone_VSSM':
            # VMamba pretrained weights are wrapped in 'model' key
            import torch
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # Load with strict=False to allow partial loading
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            loaded = len(state_dict) - len(unexpected)
            print(f'[VMamba] Successfully loaded {loaded}/{len(state_dict)} weights')
            if missing:
                print(f'[VMamba] Missing keys: {len(missing)}')
            if unexpected:
                print(f'[VMamba] Unexpected keys: {len(unexpected)}')
        else:
            # Other backbones - use standard checkpoint loading
            load_checkpoint(self.backbone, pretrained, map_location='cpu', strict=False, logger='current')

        
    def forward(self, x):
        segout = self.backbone(x)
        # Note: Channel values depend on backbone type
        # SegFormer: [64, 128, 320, 512]
        # VMamba:    [96, 192, 384, 768]
        x1 = segout[0]  # c1 x 88x88
        x2 = segout[1]  # c2 x 44x44
        x3 = segout[2]  # c3 x 22x22
        x4 = segout[3]  # c4 x 11x11

        decoder_1 = self.decode_head.forward([x1, x2, x3, x4]) # 88x88
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4)
        # cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, self.c4, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(x3)
        # cfp_out_2 += x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2 += cfp_out_2
        aa_atten_2_o = decoder_3_ra.expand(-1, self.c3, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2)
        # cfp_out_3 += x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_4_ra.expand(-1, self.c2, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1