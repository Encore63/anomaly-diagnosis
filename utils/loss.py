import torch
import numpy as np


@torch.jit.script
def tsallis_entropy(predictions: torch.Tensor, alpha=2.0, reduction='mean'):
    if alpha == 1.0:
        epsilon = 1e-8
        H = -predictions * torch.log(predictions + epsilon)
        H = H.sum(dim=1)
    else:
        H = 1 / (alpha - 1) * (1 - torch.sum(predictions ** alpha, dim=-1))
    if reduction == 'mean':
        return H.mean()
    elif reduction == 'sum':
        return H.sum()
    else:
        return H
