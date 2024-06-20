import einops
import torch
import warnings

from torch import nn
from torchinfo import summary


class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channels=10):
        super(CNN, self).__init__()
        if pretrained:
            warnings.warn("Pretrained model is not available")
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x


if __name__ == '__main__':
    from algorithms import divtent
    data = torch.rand((8, 10, 50))
    model = CNN(in_channels=10)
    domain_idx = {'source': [0, 1, 2, 3],
                  'target': [4, 5, 6, 7]}
    divtent.replace_bn_layer(model, domain_idx)
    print(model(data).shape)
    # summary(model, input_data=data)
