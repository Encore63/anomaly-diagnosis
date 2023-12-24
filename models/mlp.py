import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, norm_reduce: bool):
        super(MLP, self).__init__()
        self.norm_reduce = norm_reduce
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=out_features)
        )

    def forward(self, x):
        output = self.mlp(x)
        if self.norm_reduce:
            output = torch.norm(output)
        return output
