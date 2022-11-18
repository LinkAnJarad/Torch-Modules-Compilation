import torch
from torch import nn
from .TransformerEncoderLayer import TransformerEncoderLayer

class RefinementBlock(nn.Module):
  '''
    Implementation of the refinement block in "Learning Token-Based Representation for Image Retrieval" (https://arxiv.org/pdf/2112.06159.pdf)
    A module for refining the visual tokens outputted by the tokenizer in the paper.
    
    Parameters:
    d_model (int): dimensionality/channels of input
    nhead (int): number of attention heads in the transformer
    dim_feedforward (int): number of hidden dimensions in the feedforward layers
    dropout (int): dropout rate
    
    Returns:
    Tuple containing:
        Tensor of shape (batch_size, sequence length, d_model); same as input
  '''
  def __init__(self, d_model, nhead, dim_feedforward, dropout):
    super().__init__()
    self.self_att = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout) # Batch first
    self.cross_att = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout) # Batch first

  def forward(self, visual_tokens, cnn_output):
    '''
    Parameters:
    visual_tokens (tensor): first output of LFSATokenizer; visual tokens
    cnn_output (tensor): second output of LFSATokenizer; original, flattened (along spatial dim) input of LFSATokenizer
    '''
    self_att = self.self_att(visual_tokens, visual_tokens, visual_tokens)
    cross_att = self.cross_att(self_att, cnn_output, cnn_output)
    return cross_att
