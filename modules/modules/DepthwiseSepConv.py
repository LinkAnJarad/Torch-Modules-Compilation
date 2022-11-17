import torch
from torch import nn

class DepthwiseSepConv(nn.Module):
    '''
    From https://arxiv.org/pdf/1704.04861v1.pdf
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm_nonlinear=True):
        super(DepthwiseSepConv, self).__init__()
        
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        
        self.norm_nonlinear = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        ) if norm_nonlinear else nn.Identity()
        
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.norm_nonlinear(x)
        x = self.pointwise_conv(x)
        return x
