import torch
from torch import nn

class ZNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.max_pool = nn.MaxPool1d(dim)
        self.avg_pool = nn.AvgPool1d(dim)
        
    def forward(self, x):
        N, D, X, Y = x.shape
        x = x.reshape(N, D, -1).permute(0, 2, 1) # N, XY, D
        max_pooled = self.max_pool(x).reshape(N, 1, X, Y) 
        avg_pooled = self.avg_pool(x).reshape(N, 1, X, Y)
        
        return torch.cat((max_pooled, avg_pooled), dim=1)
      
class Branch(nn.Module):
    def __init__(self, dims, kernel_size):
        super().__init__()
        
        self.branch = nn.Sequential(
            ZNorm(dims),
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.branch(x)
      
class TripletAttention(nn.Module):
    '''
    Computes attention in a feature map across all three dimensions (channel and both spatial dims). From https://arxiv.org/pdf/2010.03045v2.pdf

    Parameters
    in_channels (int): Number of input channels
    height (int): height of feature map
    width (int): width of feature map
    kernel_size (int): kernel size of the convolutions. Default: 7
    
    Returns
    Tensor of shape (batch_size, in_channels, height, width), same as input
    
    '''
    def __init__(self, in_channels, height, width, kernel_size=7):
        super().__init__()
        
        self.hw_branch = Branch(in_channels, kernel_size)
        self.cw_branch = Branch(height, kernel_size)
        self.ch_branch = Branch(width, kernel_size)
        
    def forward(self, x):
        hw_branch_out = self.hw_branch(x) * x # N, C, H, W
        cw_branch_out = self.cw_branch(x.permute(0,2,1,3)) * x.permute(0,2,1,3) # N, H, C, W
        ch_branch_out = self.ch_branch(x.permute(0,3,2,1)) * x.permute(0,3,2,1) # N, W, H, C
        
        cw_branch_out = cw_branch_out.permute(0,2,1,3)
        ch_branch_out = ch_branch_out.permute(0,3,2,1)
        
        avg = (hw_branch_out + cw_branch_out + ch_branch_out)/3.
        return avg
