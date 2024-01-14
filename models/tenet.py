import torch

from torch import nn
from torchinfo import summary
from collections import OrderedDict
from models.attention_module import AttentionModule
from models.adversarial_module import AdversarialModule
from models.self_supervised_module import SelfSupervisedModule


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
            OrderedDict([
                ('conv_1', nn.Conv2d(in_channels=1, out_channels=self.F1,
                                     kernel_size=(1, 3), stride=1, padding=(0, 1))),
                ('bn_1', nn.BatchNorm2d(num_features=self.F1)),
                # depth-wise convolution
                ('conv_2', nn.Conv2d(in_channels=self.F1, out_channels=self.D * self.F1,
                                     kernel_size=1, groups=self.D)),
                ('bn_2', nn.BatchNorm2d(num_features=self.D * self.F1)),
                ('elu_1', nn.ELU()),
                ('avg_pool_1', nn.AvgPool2d(kernel_size=(1, 2), stride=2)),
                # ('dropout_1', nn.Dropout(p=0.5))
            ])
        )
        self.ext_block_2 = nn.Sequential(
            OrderedDict([
                # separable convolution
                ('conv_3', nn.Conv2d(in_channels=self.D * self.F1, out_channels=self.F2, kernel_size=1)),
                ('bn_3', nn.BatchNorm2d(num_features=self.F2)),
                ('elu_2', nn.ELU()),
                ('avg_pool_2', nn.AvgPool2d(kernel_size=(1, 2), stride=1)),
                # ('dropout_2', nn.Dropout(p=0.5))
            ])
        )
        self.class_classifier = nn.Sequential(
            OrderedDict([
                ('fc_1', nn.Linear(in_features=3840, out_features=1920)),
                ('elu_3', nn.ELU()),
                ('fc_2', nn.Linear(in_features=1920, out_features=960)),
                ('elu_4', nn.ELU()),
                ('fc_3', nn.Linear(in_features=960, out_features=num_classes))
            ])
        )
        self.self_supervised_branch = SelfSupervisedModule()
        self.adversarial_branch = AdversarialModule(in_features=3840, hidden_dims=[1920, 960, num_classes])

    def forward(self, x):
        features = self.ext_block_1(x)
        features = self.ext_block_2(features)
        features = features.view(features.shape[0], -1)
        output = self.class_classifier(features)
        return output


class ReTENet(nn.Module):
    """
    Modified from TENet
    """
    def __init__(self, f1: int, f2: int, depth: int, num_classes: int):
        super(ReTENet, self).__init__()
        self.F1 = f1
        self.F2 = f2
        self.D = depth
        self.ext_block_1 = nn.Sequential(
            OrderedDict([
                ('conv_1', nn.Conv2d(in_channels=1, out_channels=self.F1,
                                     kernel_size=(3, 1), stride=1, padding=(1, 0))),
                ('bn_1', nn.BatchNorm2d(num_features=self.F1)),
                # depth-wise convolution
                ('conv_2', nn.Conv2d(in_channels=self.F1, out_channels=self.D * self.F1,
                                     kernel_size=1, groups=self.D)),
                ('bn_2', nn.BatchNorm2d(num_features=self.D * self.F1)),
                ('relu_1', nn.ReLU()),
                ('avg_pool_1', nn.AvgPool2d(kernel_size=(2, 1), stride=2)),
                ('dropout_1', nn.Dropout(p=0.5))
            ])
        )
        self.ext_block_2 = nn.Sequential(
            OrderedDict([
                # separable convolution
                ('conv_3', nn.Conv2d(in_channels=self.D * self.F1, out_channels=self.F2, kernel_size=1)),
                ('bn_3', nn.BatchNorm2d(num_features=self.F2)),
                ('relu_2', nn.ReLU()),
                ('avg_pool_2', nn.AvgPool2d(kernel_size=(2, 1), stride=1)),
                ('dropout_2', nn.Dropout(p=0.5))
            ])
        )
        self.class_classifier = nn.Sequential(
            OrderedDict([
                ('fc_1', nn.Linear(in_features=3200, out_features=1600)),
                ('relu_3', nn.ReLU()),
                ('fc_2', nn.Linear(in_features=1600, out_features=800)),
                ('relu_4', nn.ReLU()),
                ('fc_3', nn.Linear(in_features=800, out_features=num_classes))
            ])
        )
        self.self_supervised_branch = SelfSupervisedModule()
        self.adversarial_branch = AdversarialModule(in_features=3840, hidden_dims=[1920, 960, num_classes])
        self.attention_block = AttentionModule(embed_dim=32, num_heads=4)

    def forward(self, x):
        features = self.ext_block_1(x)
        features = self.ext_block_2(features)
        features = features.reshape(features.shape[0], -1)
        output = self.class_classifier(features)
        return output


if __name__ == '__main__':
    data = torch.randn((32, 1, 10, 50))
    model = ReTENet(f1=16, f2=32, depth=8, num_classes=10)
    out = model(data)
    summary(model, input_data=data)
