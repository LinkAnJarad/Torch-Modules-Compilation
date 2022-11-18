import torch
from torch import nn

class GlobalContextModule(nn.Module):
    '''
    A sort of self-attention (non-local) block on feature maps.
    Implementation of :GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond" (https://arxiv.org/pdf/1904.11492.pdf)
    
    Parameters:
    input_channels (int): Number of input channels
    
    Returns:
    Tensor of shape (batch_size, channels, height, width); same as input
    '''
    def __init__(self, input_channels):
        super().__init__()
        
        self.context_modeling = nn.Conv2d(input_channels, input_channels, 1, 1)
        self.transform = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 1, 1),
            nn.GroupNorm(1, input_channels), # group norm with single channel is equivalent to layernorm
            nn.SiLU(),
            nn.Conv2d(input_channels, input_channels, 1, 1)
        )
        
    def forward(self, x):
        att = x * self.context_modeling(x).softmax(1)
        x = x + self.transform(att)
        return x
