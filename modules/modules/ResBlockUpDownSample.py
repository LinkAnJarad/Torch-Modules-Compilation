import torch
from torch import nn

class BasicConv(nn.Module):
    def __init__(self, num_channels, num_groups, dropout):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups, num_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, num_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return x + self.conv(x)
    

class ResBlockUpDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, num_layers, dropout, sample='down'):
        super().__init__()
        
        self.conv_modules = nn.ModuleList([BasicConv(in_channels, num_groups, dropout) for _ in range(num_layers)])
        assert sample in ['down', 'up', 'none']
        if sample == 'down':
            self.sample = nn.Conv2d(in_channels, out_channels, 4, 2, padding=1)
        elif sample == 'up':
            self.sample = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding=1)
        else:
            self.sample = nn.Identity()
            
    def forward(self, x):
        for block in self.conv_modules:
            x = block(x)
        return self.sample(x)
