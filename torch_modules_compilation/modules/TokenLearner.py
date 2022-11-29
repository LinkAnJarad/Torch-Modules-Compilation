import torch
from torch import nn

class TokenLearner(nn.Module):
    def __init__(self, in_channels, num_tokens):
        super().__init__()
        
        self.selection = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, num_tokens, 3, 1, padding='same', bias=False),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, 3, 1, padding='same', bias=False),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, 3, 1, padding='same', bias=False),
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, 3, 1, padding='same', bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention = self.selection(x) 
        attention = attention.unsqueeze(2)
        x = x.unsqueeze(1)
        x = x * attention
        
        N, S, C, H, W = x.shape
        x = x.reshape(N, S, C, -1)
        x = x.mean(-1).squeeze()
        
        return x
