import torch
from torch import nn

      
class PatchMerger(nn.Module):
    '''
    Copied from lucidrains' repo https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_with_patch_merger.py under the MIT license.
    From "LEARNING TO MERGE TOKENS IN VISION TRANSFORMERS" (https://arxiv.org/pdf/2202.12015v1.pdf)
    
    Merges N tokens into M tokens in transformer models. Typically added in-between transformer layers.
    
    Parameters:
    dim (int): dimensionality/channels of the tokens
    output_tokens (int): number of output merged tokens
    norm (bool): normalize the input before merging
    scale (bool): scale the attention matrix by the square root of dim (for numerical stability)
    
    Returns:
    Tensor of shape (batch_size, output_tokens, dim)
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
