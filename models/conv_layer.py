"""
Basic convolution layers for refinement modules.
Copied from ColonFormer lib/conv_layer.py
"""
import torch
import torch.nn as nn
class Conv(nn.Module):
    """Basic Conv + BN + ReLU layer"""
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        
        self.bn_acti = bn_acti
        
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        
        if self.bn_acti:
            self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
            self.act = nn.PReLU(nOut)
    
    def forward(self, x):
        output = self.conv(x)
        
        if self.bn_acti:
            output = self.bn(output)
            output = self.act(output)
            
        return output
class BNPReLU(nn.Module):
    """BatchNorm + PReLU"""
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.act = nn.PReLU(nIn)
    
    def forward(self, x):
        return self.act(self.bn(x))