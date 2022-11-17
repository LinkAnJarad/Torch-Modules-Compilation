import torch
from torch import nn

class FeatureMapAttention(nn.Module):
    '''
    Adapted from https://github.com/rosinality/sagan-pytorch/blob/master/model.py#L82
    '''
    def __init__(self, in_channels):
        super(FeatureMapAttention, self).__init__()
        
        self.qw = nn.Conv1d(in_channels, in_channels, 1, 1)
        self.kw = nn.Conv1d(in_channels, in_channels, 1, 1)
        self.vw = nn.Conv1d(in_channels, in_channels, 1, 1)
        
        self.gamma = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, query, key, value):
        shape = query.shape
        flatten_q = query.view(shape[0], shape[1], -1)
        flatten_k = key.view(shape[0], shape[1], -1)
        flatten_v = value.view(shape[0], shape[1], -1)

        query = self.qw(flatten_q).permute(0, 2, 1)
        key = self.kw(flatten_k)
        value = self.vw(flatten_v)
        query_key = torch.bmm(query, key) / shape[-1]
        attn = query_key.softmax(dim=1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        value = value.view(*shape)
        out = self.gamma * attn + value
        return out
