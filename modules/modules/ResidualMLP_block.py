import torch
from torch import nn

class ResidualMLP_block(nn.Module):
    '''
    Adapted from https://arxiv.org/pdf/2108.08186.pdf
    '''
    def __init__(self, dim, ic_first=True, dropout=0.):
        super().__init__()
        
        self.ic_first = ic_first
        self.ic1 = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )
        self.ic2 = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
    
    def forward(self, x):
        if self.ic_first:
            y = self.ic1(x)
            y = self.linear1(y)
        else:
            y = self.linear1(x)
        y = nn.SiLU()(y)
        y = self.ic2(y)
        y = self.linear2(y)
        y = y + x
        y = nn.SiLU()(y)
        return y
