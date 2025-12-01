"""
UPerNet Decoder - Simplified from ColonFormer
Adapted for ColonMamba with channel projection support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class PPM(nn.Module):
    """Pyramid Pooling Module"""
    def __init__(self, in_channels, out_channels=64, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(out_channels // len(pool_sizes), eps=1e-3),
                nn.ReLU(inplace=True)
            )
            for pool_size in pool_sizes
        ])
        
        # Bottleneck
        total_ch = in_channels + out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_ch, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        pyramids = [x]
        
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=True)
            pyramids.append(upsampled)
        
        output = torch.cat(pyramids, dim=1)
        output = self.bottleneck(output)
        return output
class UPerHead(nn.Module):
    """
    UPerNet Decoder Head - Matches ColonFormer decode_head
    
    Args:
        in_channels: List of encoder output channels [64, 128, 320, 512]
        channels: Unified decoder channels (default: 64)
        num_classes: Number of output classes
    """
    def __init__(self, in_channels=[64, 128, 320, 512], channels=64, num_classes=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        
        # PSP Module on deepest features
        self.psp_modules = PPM(
            in_channels=in_channels[-1],  # 512
            out_channels=channels,  # 64
            pool_sizes=[1, 2, 3, 6]
        )
        
        # Lateral convolutions for FPN
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_ch in in_channels[:-1]:  # [64, 128, 320]
            # Lateral conv: in_ch -> channels
            lateral = nn.Sequential(
                nn.Conv2d(in_ch, channels, 1, bias=False),
                nn.BatchNorm2d(channels, eps=1e-3),
                nn.ReLU(inplace=True)
            )
            
            # FPN conv
            fpn_conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels, eps=1e-3),
                nn.ReLU(inplace=True)
            )
            
            self.lateral_convs.append(lateral)
            self.fpn_convs.append(fpn_conv)
        
        # FPN bottleneck
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)
    
    def forward(self, inputs):
        """
        Args:
            inputs: List [x1, x2, x3, x4]
        
        Returns:
            Tensor: [B, num_classes, H/4, W/4]
        """
        # PSP forward on deepest
        x = inputs[-1]  # x4
        
        # Apply PPM stages
        psp_outs = [x]
        for stage in self.psp_modules.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=True)
            psp_outs.append(upsampled)
        
        psp_outs = torch.cat(psp_outs, dim=1)
        psp_out = self.psp_modules.bottleneck(psp_outs)
        
        # Build laterals
        laterals = [self.lateral_convs[i](inputs[i]) for i in range(len(self.lateral_convs))]
        laterals.append(psp_out)
        
        # Top-down fusion
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=True
            )
        
        # FPN outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(self.fpn_convs))]
        fpn_outs.append(laterals[-1])
        
        # Resize all to same size
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size, mode='bilinear', align_corners=True)
        
        # Concatenate and fuse
        output = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(output)
        output = self.cls_seg(output)
        
        return output