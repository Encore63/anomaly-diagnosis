import torch

from torch import nn
from torchinfo import summary


class SelfSupervisedModule(nn.Module):
    def __init__(self):
        super(SelfSupervisedModule, self).__init__()
        ...

    def forward(self, x):
        ...


if __name__ == '__main__':
    ...
