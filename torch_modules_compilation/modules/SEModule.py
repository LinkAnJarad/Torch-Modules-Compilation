import torch
from torch import nn

class SEModule(nn.Module):
    '''
    Module that computes channel-wise interactions in a feature map. From https://arxiv.org/pdf/1709.01507v4.pdf
    
    Parameters
    
    in_channels (int): Number of input channels
    reduced_channels (int): Number of channels to reduce to in the "squeeze" part of the module
    feature_map_size (int): height/width of the feature map
    
    Returns:
    Tensor of shape (batch_size, channels, height, width), same as input
    '''
    def __init__(self, in_channels, reduced_channels, feature_map_size):
        super().__init__()
        
        self.GAP = nn.AvgPool2d(feature_map_size)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, in_channels)
        )
        
    def forward(self, x):
        att = self.GAP(x).squeeze()
        N, C = att.shape
        att = self.ffn(att).reshape(N, C, 1, 1).sigmoid()
        return att * x
