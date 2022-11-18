import torch
from torch import nn

class ResidualMLP_downsample(nn.Module):
    '''
    Adapted from "Generalizing MLPs With Dropouts, Batch Normalization, and Skip Connections" (https://arxiv.org/pdf/2108.08186.pdf)
    An improvement of standard MLPs along with residual connections. This implements the downsampling MLP block (eq. 6 in the paper)
    
    Parameters:
    dim (int): number of input dimensions
    downsample_dim (int): number of output dimensions
    dropout (float): dropout rate
    
    Returns:
    Tensor of shape (batch_size, downsample_dim)
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
