import torch

from torch import nn
from torchinfo import summary


class Encoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True))

        self.conv_4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True))

        self.fc_1 = nn.Linear(16 * 16 * 32, 256)
        self.fc_2 = nn.Linear(256, out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        noise = torch.rand(x.shape) * x.mean() / 10
        x = x + noise.cuda()
        output = self.conv_1(x)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = output.view(output.size(0), -1)
        h1 = self.relu(self.fc_1(output))
        return self.fc_2(h1)


class Decoder(nn.Module):
    def __init__(self, in_channel: int):
        super(Decoder, self).__init__()
        self.fc_3 = nn.Linear(in_channel, 256)
        self.fc_4 = nn.Linear(256, 8192)
        self.relu = nn.ReLU()

        self.trans_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True))

        self.trans_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True))

        self.trans_conv_3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True))

        self.trans_conv_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True))

    def forward(self, x):
        h3 = self.relu(self.fc_3(x))
        output = self.relu(self.fc_4(h3))
        output = output.view(output.size(0), 32, 16, 16)
        output = self.trans_conv_1(output)
        output = self.trans_conv_2(output)
        output = self.trans_conv_3(output)
        output = self.trans_conv_4(output)
        return output


class Classifier(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(Classifier, self).__init__()
        self.fc_5 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_channel, out_channel)
        )

    def forward(self, x):
        label = self.fc_5(x)
        return label


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(DenoisingAutoEncoder, self).__init__()
        self.encoder = Encoder(in_channel, out_channel)
        self.decoder = Decoder(out_channel)

    def forward(self, x):
        embedding = self.encoder(x)
        # output = self.decoder(embedding)
        return embedding


if __name__ == '__main__':
    data = torch.rand((1, 1, 10, 50)).cuda()
    model = DenoisingAutoEncoder(in_channel=1, out_channel=1)
    out = model(data).cuda()
    print(out.shape)
    # summary(model, input_data=data)
