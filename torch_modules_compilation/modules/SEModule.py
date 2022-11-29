import torch
from torch import nn

class SEModule(nn.Module):
    '''
    https://arxiv.org/pdf/1709.01507v4.pdf
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
