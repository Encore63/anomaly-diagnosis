import torch
import warnings

from torch import nn
from models.cnn import CNN
from models.mrfgcn import MrfGCN
from models.resnet import resnet
from einops import rearrange
from torchinfo import summary
from einops import rearrange, reduce, repeat


class DAGCN(nn.Module):
    def __init__(self, in_channels=10, num_classes=10, pretrained=False):
        super(DAGCN, self).__init__()
        self.model_cnn = CNN(in_channels=in_channels)
        self.model_GCN = MrfGCN(pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        x = rearrange(x, 'B L E -> B E L')
        x1 = self.model_cnn(x)
        x2 = self.model_GCN(x1)
        output = self.classifier(x2)
        return output


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data = torch.rand((32, 50, 30)).cuda()
    model = DAGCN(in_channels=50).cuda()
    summary(model, input_data=data)
    # print(model(data).shape)
