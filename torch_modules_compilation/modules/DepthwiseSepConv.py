import torch
from torch import nn

class DepthwiseSepConv(nn.Module):
    '''
    A depthwise seperable convolution; consists of a depthwise convolution and a pointwise convolution. Used in MobileNets and used in the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
    Applications" (https://arxiv.org/pdf/1704.04861v1.pdf).
    
    Parameters: 
    in_channels (int): Number of input channels
    out_channels (int): Number of output channels
    kernel_size (int): Size of depthwise convolution kernel
    stride (int): Stride of depthwise convolution
    
    Returns:
    Tensor of shape (batch_size, channels, height, width)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm_nonlinear=True):
        super(DepthwiseSepConv, self).__init__()
        
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nn.ReLU()
        
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.nonlinearity(self.norm1(x))
        x = self.pointwise_conv(x)
        x = self.nonlinearity(self.norm2(x))
        return x
