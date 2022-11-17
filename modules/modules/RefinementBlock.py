import torch
from torch import nn
from .TransformerEncoderLayer import TransformerEncoderLayer

class RefinementBlock(nn.Module):
  '''
  Learning Token-Based Representation for Image Retrieval
  '''
  def __init__(self, d_model, nhead, dim_feedforward, dropout):
    super().__init__()
    self.self_att = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout) # Batch first
    self.cross_att = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout) # Batch first

  def forward(self, visual_tokens, cnn_output):
    self_att = self.self_att(visual_tokens, visual_tokens, visual_tokens)
    cross_att = self.cross_att(self_att, cnn_output, cnn_output)
    return cross_att
