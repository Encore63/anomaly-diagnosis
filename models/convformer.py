import torch

from torch import nn
from torchinfo import summary
from transformer import Transformer


class ConvFormer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvFormer, self).__init__()

        self.conv_module = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.MaxPool1d(2)
        )

        # self.transformer_layer_1 = Transformer(dim=64, depth=1, heads=8)

    def forward(self, x):
        ...
