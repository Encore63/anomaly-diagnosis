import torch

from torch import nn
from models.transformer import Transformer


class ResBase(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResBase.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * ResBase.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != ResBase.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResBase.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResBase.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, in_channels, block, num_block, out_channels=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, out_channels)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make resnet layers(by layer I did not mean this 'layer' was the
        same as a neuron network layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottleneck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class SiamNet(nn.Module):
    def __init__(self, input_dim=50, depth=12, num_heads=5, hidden_dim=128, num_classes=10, fusion_way='add'):
        super(SiamNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.fusion_way = fusion_way

        # ResNet-based block
        self.residual_block = ResNet(in_channels=1, out_channels=hidden_dim, block=ResBase, num_block=[2, 2, 2, 2])

        # Transformer-based block
        self.attention_block = Transformer(dim=input_dim, depth=depth, heads=num_heads, mlp_dim=hidden_dim)

        # Multi-layer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward_by_residual_block(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(dim=1)
        assert len(x.shape) > 3, "Invalid data dimension for residual block!"
        high_dim_outputs = self.residual_block(x)
        return high_dim_outputs

    def forward_by_attention_block(self, x):
        if len(x.shape) > 3:
            x = x.squeeze(dim=1)
        assert len(x.shape) < 4, "Invalid data dimension for attention block!"
        low_dim_outputs = self.attention_block(x)
        return low_dim_outputs

    def forward(self, x):
        high_dim_outputs = self.forward_by_residual_block(x)
        low_dim_outputs = self.forward_by_attention_block(x)

        high_dim_outputs = high_dim_outputs.view(high_dim_outputs.size(0), -1)
        low_dim_outputs = low_dim_outputs.view(low_dim_outputs.size(0), -1)

        if self.fusion_way == 'concat':
            outputs = torch.cat((high_dim_outputs, low_dim_outputs), dim=0)
        else:
            outputs = high_dim_outputs + low_dim_outputs
        outputs = self.mlp(outputs)

        return outputs


if __name__ == '__main__':
    from torchinfo import summary
    model = SiamNet()
    _x = torch.randn((16, 1, 32, 50))
    summary(model, input_data=_x)
