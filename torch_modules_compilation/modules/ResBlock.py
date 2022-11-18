import torch
from torch import nn


class ResBlock(nn.Module):
    '''
    Your basic residual block.
    
    Parameters:
    in_channels (int): number of input channels
    kernel_size (int): kernel size
    dropout (float): dropout rate
    
    Returns:
    Tensor of shape (batch_size, channels, height, width); same as input
    '''
    def __init__(self, in_channels, kernel_size, dropout):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.norm_nonlinear = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return self.norm_nonlinear(y + x)
