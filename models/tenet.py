import torch

from torch import nn
from torchinfo import summary
from models.ss_head import SSHead


class TENet(nn.Module):
    """
    Modified from EEGNet
    """
    def __init__(self, f1: int, f2: int, depth: int, num_classes: int):
        super(TENet, self).__init__()
        self.F1 = f1
        self.F2 = f2
        self.D = depth
        self.ext_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(num_features=self.F1),
            # depth-wise convolution
            nn.Conv2d(in_channels=self.F1, out_channels=self.D * self.F1, kernel_size=1, groups=self.D),
            nn.BatchNorm2d(num_features=self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=2),
            nn.Dropout(p=0.5)
        )
        self.ext_block_2 = nn.Sequential(
            # separable convolution
            nn.Conv2d(in_channels=self.D * self.F1, out_channels=self.F2, kernel_size=1),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=1),
            nn.Dropout(p=0.5)
        )
        self.ss_branch = SSHead()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        features = self.ext_block_1(x)
        features = self.ext_block_2(features)
        output = self.classifier(features.view(features.shape[0], -1, 256))
        return output


if __name__ == '__main__':
    data = torch.randn((32, 1, 10, 50))
    model = TENet(f1=64, f2=128, depth=8, num_classes=10)
    out = model(data)
    print(out.shape)
    summary(model, input_data=data)
