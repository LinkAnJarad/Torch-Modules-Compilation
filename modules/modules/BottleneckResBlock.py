import torch
from torch import nn

class BottleneckResBlock(nn.Module):
    '''
    A residual bottleneck block. From the paper "Deep Residual Learning for Image Recognition"
    https://arxiv.org/pdf/1512.03385.pdf
    
    Parameters:
    in_channels (int): number of input channels
    bottleneck_channels (int): number of bottleneck channels; usually less than the number of bottleneck channels
    dropout (float): dropout rate; performed after every convolution
    
    Returns:
    Tensor of shape (batch_size, channels, height, width)
    '''
    
    def __init__(self, in_channels, bottleneck_channels, dropout=0.):
        super(BottleneckResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, 1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(bottleneck_channels, in_channels, 1, 1)
        self.norm1 = nn.BatchNorm2d(bottleneck_channels)
        self.norm2 = nn.BatchNorm2d(bottleneck_channels)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = nn.SiLU()
        
    def forward(self, x):
        res = x
        x = self.nonlinearity(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.nonlinearity(self.norm2(self.conv2(x)))
        x = self.dropout(x)
        x = res + (self.conv3(x))
        x = self.nonlinearity(self.norm3(x))
        return x
