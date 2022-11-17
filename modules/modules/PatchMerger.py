import torch
from torch import nn

      
class PatchMerger(nn.Module):
    '''
    Copied from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_with_patch_merger.py
    Paper from https://arxiv.org/pdf/2202.12015v1.pdf
    '''
    def __init__(self, dim, output_tokens, norm=False, scale=False):
        super(PatchMerger, self).__init__()
        
        self.scale = dim ** -0.5 if scale else 1.
        self.w = nn.Parameter(torch.empty(output_tokens,dim))
        nn.init.normal_(self.w)
        
        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()
        
    def forward(self, x):
        x = self.norm(x)
        qk = torch.matmul(self.w, x.permute(0,2,1)) * self.scale
        qk = qk.softmax(dim=-1)
        qkv = torch.matmul(qk, x)
        return qkv
