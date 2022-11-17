import torch
from torch import nn

class BottleneckResBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, dropout=0.):
        super(BottleneckResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, 1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(bottleneck_channels, in_channels, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = nn.SiLU()
        
    def forward(self, x):
        res = x
        x = self.nonlinearity(self.conv1(x))
        x = self.dropout(x)
        x = self.nonlinearity(self.conv2(x))
        x = self.dropout(x)
        x = self.nonlinearity(self.conv3(x))
        x = self.dropout(x)
        x = x + res
        return self.nonlinearity(x)
