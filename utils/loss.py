import torch
import numpy as np


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


def conjugate_loss(outputs, num_classes=10, eps=6.0, temp=1.0):
    outputs = outputs / temp
    predictions = torch.softmax(outputs, dim=1)

    smax_inp = predictions

    eye = torch.eye(num_classes).to(outputs.device)
    eye = eye.reshape((1, num_classes, num_classes))
    eye = eye.repeat(outputs.shape[0], 1, 1)
    t2 = eps * torch.diag_embed(smax_inp)
    smax_inp = torch.unsqueeze(smax_inp, 2)
    t3 = eps * torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
    matrix = eye + t2 - t3
    y_star = torch.linalg.solve(matrix, smax_inp)
    y_star = torch.squeeze(y_star)

    pseudo_prob = y_star
    loss = torch.logsumexp(outputs, dim=1) - (pseudo_prob * outputs - eps * pseudo_prob * (1 - predictions)).sum(
        dim=1)

    return loss