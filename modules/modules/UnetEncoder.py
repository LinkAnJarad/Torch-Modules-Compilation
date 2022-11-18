import torch
from torch import nn

class BasicConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, dropout):
        super().__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
            nn.GroupNorm(1, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding),
            nn.GroupNorm(1, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.convs(x)
    
class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.downsampler = nn.Conv2d(in_channels, out_channels, 4, 2, padding=1)
        
    def forward(self, x):
        return self.downsampler(x)
    
class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding=1)
        
    def forward(self, x):
        return self.upsampler(x)
      
class UnetEncoder(nn.Module):
    '''
    Standard UNet encoder.
    
    Parameters:
    channels (list of ints): A list containing the number of channels in the encoder. E.g [3, 64, 128, 256]
    dropout (float): dropout rate
    
    Returns:
    Tensor of shape (batch_size, channels, height, width)
    '''
    def __init__(self, channels, dropout):
        super().__init__()
        
        self.blocks = nn.ModuleList([])
        self.downsamplers = nn.ModuleList([])
        
        for c in range(len(channels[:-1])):
            
            self.blocks.append(BasicConv(channels[c], channels[c+1], 3, 1, 1, dropout))
            self.downsamplers.append(Downsampler(channels[c+1], channels[c+1]))
            
    def forward(self, x):
        feature_maps = []
        
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            feature_maps.append(x)
            x = self.downsamplers[i](x)
            
        return feature_maps
