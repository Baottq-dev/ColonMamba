"""
Decoder Module - Original Design
Concat-based decoder matching diagram architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv block with residual connection"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x if self.residual is None else self.residual(x)
        out = self.conv(x)
        return F.relu(out + identity, inplace=True)


class PPM(nn.Module):
    """Pyramid Pooling Module"""
    
    def __init__(self, in_channels, out_channels=64, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(out_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for pool_size in pool_sizes
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        H, W = x.size(2), x.size(3)
        pyramids = [x]
        
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=True)
            pyramids.append(upsampled)
        
        out = torch.cat(pyramids, dim=1)
        out = self.bottleneck(out)
        return out


class UPerHead(nn.Module):
    """
    Decoder - Original Design
    Concat-based decoder from diagram (1/4 resolution concat)
    
    NOTE: Class name kept as "UPerHead" for compatibility,
    but implementation is completely different (concat-based design)
    """
    
    def __init__(self, in_channels=[64, 128, 320, 512], channels=64, num_classes=1):
        super().__init__()
        
        self.in_channels = in_channels
        
        total_channels = sum(in_channels)  # 1024
        
        # Conv blocks
        self.conv_block1 = ConvBlock(total_channels, 512)
        self.conv_block2 = ConvBlock(512, 256)
        self.conv_block3 = ConvBlock(256, 128)
        self.conv_block4 = ConvBlock(128, 128)
        
        # PPM
        self.ppm = PPM(in_channels=128, out_channels=64)
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, inputs):
        """
        Args:
            inputs: List [F1, F2, F3, F4]
        Returns:
            mask: [B, num_classes, H/4, W/4]
        """
        F1, F2, F3, F4 = inputs
        
        # Dynamic target size from F1 (supports multi-scale training)
        target = F1.shape[2]  # Use F1's spatial size
        
        # Resize all to F1's resolution
        F1_resized = F1
        F2_resized = F.interpolate(F2, size=(target, target), mode='bilinear', align_corners=True)
        F3_resized = F.interpolate(F3, size=(target, target), mode='bilinear', align_corners=True)
        F4_resized = F.interpolate(F4, size=(target, target), mode='bilinear', align_corners=True)
        
        # Concat
        x = torch.cat([F1_resized, F2_resized, F3_resized, F4_resized], dim=1)
        
        # Conv blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # PPM
        x = self.ppm(x)
        
        # Prediction
        mask = self.head(x)
        
        return mask