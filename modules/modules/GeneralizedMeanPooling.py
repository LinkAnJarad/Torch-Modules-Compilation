import torch
from torch import nn

class GeneralizedMeanPooling(nn.Module):
    '''Copied and Modified from https://github.com/feymanpriv/DOLG-paddle/blob/10653be915292c6c83e899706931e6c885a7f8aa/model/dolg_model.py#L50
    - At p = infinity, one gets Max Pooling
    - At p = 1, one gets Average Pooling
    - p can be trainable parameter
    '''
    def __init__(self, norm, output_size=1, eps=1e-6, input_dim=2):
        super(GeneralizedMeanPooling, self).__init__()
        
        assert norm > 0
        self.p = float(norm)
        self.eps = eps
        if input_dim == 1:
            self.avg_pooling = nn.AdaptiveAvgPool1d(output_size)
        elif input_dim == 2:
            self.avg_pooling = nn.AdaptiveAvgPool2d(output_size)
        elif input_dim == 3:
            self.avg_pooling = nn.AdaptiveAvgPool3d(output_size)
    
    def forward(self, x):
        x = torch.clip(x, min=self.eps).pow(self.p)
        x = self.avg_pooling(x).pow(1./self.p)
        return 
