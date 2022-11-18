import torch 
from torch import nn

class TransformerEncoderLayer(nn.Module):
    '''
    Standard transformer encoder layer with queries, keys, and values as inputs.
    
    Parameters:
    d_model (int): model dimensionality
    nhead (int): number of attention heads
    dim_feedforward (int): number of hidden dimensions in the feedforward layers
    dropout (float): dropout rate
    kdim (int, optional): dimensions of the keys
    vdim (int, optional): dimensions of the values
    
    Returns:
    Tensor of shape (batch_size, seq_len_of_queries, dim_of_queries); Same shape as queries
    '''
    def __init__(self, d_model, nhead, dim_feedforward, dropout, kdim=None, vdim=None):
        super().__init__()
        
        if kdim == None:
            kdim = d_model
        if vdim == None:
            vdim = d_model
        
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout, kdim=kdim, vdim=vdim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.att_dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.mha(query, key, value, attn_mask=mask)
        attn_output = self.att_dropout(attn_output)
        attn_output = self.norm1(query + attn_output)
        ffn_output = self.ffn(attn_output)
        ffn_output =  self.norm2(ffn_output + attn_output)
        return ffn_output
