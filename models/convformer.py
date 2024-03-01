import torch

from torch import nn
from torchinfo import summary
from .transformer import Transformer
from einops import rearrange, repeat


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()

        self.conv_module = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2)
        )

        self.transformer = nn.Sequential(
            Transformer(dim=64, depth=1, heads=8, mlp_dim=128),
            Transformer(dim=64, depth=1, heads=8, mlp_dim=128)
        )

        self.linear_module = nn.Sequential(
            nn.Linear(in_features=64 * 4, out_features=32 * 4),
            nn.Linear(in_features=32 * 4, out_features=16 * 4)
        )

    def forward(self, x):
        features = self.conv_module(x)
        features = rearrange(features, 'b c h -> b h c')
        features = self.transformer(features)
        features = features.view(features.size(0), -1)
        output = self.linear_module(features)
        return output


class Classifier(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=num_classes)
        )

    def forward(self, x):
        output = self.classifier(x)
        return output


class ConvFormer(nn.Module):
    def __init__(self, in_channels=10, hidden_dim=32, num_classes=10):
        super(ConvFormer, self).__init__()

        self.feat_ext = FeatureExtractor(in_channels)
        self.classifier = Classifier(in_channels=64, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x):
        output = self.feat_ext(x)
        output = self.classifier(output)
        return output


if __name__ == '__main__':
    classifier = Classifier(in_channels=64, hidden_dim=128, num_classes=10)
    model = FeatureExtractor(in_channels=10)

    data = torch.rand((1, 10, 50))
    logit = model(data)
    logit = classifier(logit)
    print(logit.shape)
