import torch

from torch import nn
from einops import rearrange
from torchinfo import summary


class AttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(AttentionModule, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = rearrange(x, 'c h w -> h w c')
        output, _ = self.attention_layer(x, x, x)
        output = self.relu(output)
        output, _ = self.attention_layer(output, output, output)
        output = self.relu(output)
        output = rearrange(output, 'h w c -> c h w')
        return output


if __name__ == '__main__':
    t = torch.randn((1024, 10, 50))
    model = AttentionModule(embed_dim=1024, num_heads=8)
    summary(model, input_data=t)
