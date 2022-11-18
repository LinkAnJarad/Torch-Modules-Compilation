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
    '''
    Composed of several residual blocks and a down/up sampling at the end; adapted from Stable Diffusion's ResnetBlock (https://github.com/CompVis/stable-diffusion/blob/ce05de28194041e030ccfc70c635fe3707cdfc30/ldm/modules/diffusionmodules/model.py#L82)
    
    Parameters:
    in_channels (int): number of input channels
    out_channels (int): number of output channels
    num_groups (int): number of groups for Group Normalization
    num_layers (int): number of residual blocks
    dropout (float): dropout rate
    sample (str): One of "down", "up", or "none". For downsampling 2x, use "down". For upsampling 2x, use "up". Use "none" for no down/up sampling.
    
    Returns:
    Tensor of shape (batch_size, channels, height, width)
    '''
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
