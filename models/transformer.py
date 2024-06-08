import torch
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.attend = nn.MultiheadAttention(dim, heads, dropout=dropout)

    def forward(self, x):
        q = x.permute((1, 0, 2))
        k = x.permute((1, 0, 2))
        v = x.permute((1, 0, 2))
        out, _ = self.attend(q, k, v)
        out = out.permute((1, 0, 2))
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., use_pool=False):
        super().__init__()
        self.use_pool = use_pool
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.pool = nn.AdaptiveAvgPool2d((2, mlp_dim // 2))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        if self.use_pool:
            x = self.pool(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary

    model = Transformer(dim=50, depth=6, heads=10, mlp_dim=128, dropout=0.2)
    data = torch.randn((128, 32, 50))
    summary(model, input_data=data)
