import torch
from torch import nn

class LFSATokenizer(nn.Module):
    '''
    Learning Token-Based Representation for Image Retrieval
    '''
    def __init__(self, in_channels, num_att_maps):
        super().__init__()
        
        self.L = num_att_maps
        self.spatial_att_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_att_maps, 1, 1),
            nn.BatchNorm2d(num_att_maps),
            nn.SiLU(),
            nn.Conv2d(num_att_maps, num_att_maps, 1, 1)
        )
        
    def forward(self, x):
        N, C, H, W = x.shape
        flattened_input = x.reshape(N, C, int(H*W))
        spatial_att = self.spatial_att_conv(x).reshape(N, self.L, int(H*W))
        spatial_att = spatial_att.softmax(dim=-1)
        visual_tokens = torch.bmm(spatial_att, flattened_input.permute(0, 2, 1)) #/ spatial_att.sum(-1, keepdim=True)
        return visual_tokens, flattened_input.permute(0, 2, 1)
