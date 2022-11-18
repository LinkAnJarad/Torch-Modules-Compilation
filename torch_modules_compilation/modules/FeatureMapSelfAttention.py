import torch
from torch import nn

class FeatureMapSelfAttention(nn.Module):
    '''
    Copied from https://github.com/rosinality/sagan-pytorch/blob/master/model.py#L82 under Apache 2.0 License.
    A feature map self-attention module used in SAGAN; "Self-Attention Generative Adversarial Networks" (https://arxiv.org/pdf/1805.08318.pdf)
    
    Parameters:
    in_channels (int): Number of input channels
    
    Returns:
    Tensor of shape (batch_size, channels, height, width); same shape as input
    '''
    def __init__(self, in_channels):
        super(FeatureMapSelfAttention, self).__init__()
        
        self.qw = nn.Conv1d(in_channels, in_channels, 1, 1)
        self.kw = nn.Conv1d(in_channels, in_channels, 1, 1)
        self.vw = nn.Conv1d(in_channels, in_channels, 1, 1)
        
        self.gamma = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        shape = x.shape
        flatten = x.view(shape[0], shape[1], -1)
        query = self.qw(flatten).permute(0, 2, 1)
        key = self.kw(flatten)
        value = self.vw(flatten)
        query_key = torch.bmm(query, key) / shape[-1]
        attn = query_key.softmax(dim=1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + x
        return out
