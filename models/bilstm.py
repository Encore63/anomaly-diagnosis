import torch

from torch import nn
from torchinfo import summary


class BiLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(BiLSTM, self).__init__()
        self.hidden_dim = 64
        self.kernel_num = 16
        self.num_layers = 2
        self.V = 5

        self.ext_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, self.kernel_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.kernel_num),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.ext_block_2 = nn.Sequential(
            nn.Conv2d(self.kernel_num, self.kernel_num * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.kernel_num * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(self.V))

        self.fc_1 = nn.Sequential(nn.Linear(self.V * self.V * 2 * self.hidden_dim, self.hidden_dim * 4),
                                  nn.ReLU(), nn.Dropout())

        self.fc_2 = nn.Linear(self.hidden_dim * 4, out_channel)

        self.bi_lstm = nn.LSTM(self.kernel_num * 2, self.hidden_dim,
                               num_layers=self.num_layers, bidirectional=True,
                               batch_first=True, bias=False)

    def forward(self, x):
        x = self.ext_block_1(x)
        x = self.ext_block_2(x)
        x = x.view(-1, self.kernel_num * 2, self.V * self.V)
        x = torch.transpose(x, 1, 2)
        bi_lstm_out, _ = self.bi_lstm(x)
        bi_lstm_out = torch.tanh(bi_lstm_out)
        bi_lstm_out = bi_lstm_out.reshape(bi_lstm_out.size(0), -1)
        logit = self.fc_1(bi_lstm_out)
        logit = self.fc_2(logit)

        return logit


if __name__ == '__main__':
    data = torch.randn((1, 1, 50, 50))
    model = BiLSTM(out_channel=10)
    out = model(data)
    summary(model, input_data=data)
