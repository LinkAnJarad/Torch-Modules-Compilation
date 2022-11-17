import torch
from torch import nn

class ResidualMLP_downsample(nn.Module):
    '''
    Adapted from https://arxiv.org/pdf/2108.08186.pdf
    '''
    def __init__(self, dim, downsample_dim, dropout):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, downsample_dim),
            nn.SiLU()
        )
        
    def forward(self, x):
        return self.downsample(x)
