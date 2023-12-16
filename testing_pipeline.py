import torch

from utils.data_utils import DataTransform


def test(test_iter, model_path, criterion):
    model = torch.load(model_path)
    model.eval()
    count = 0
    for _, (data, label) in enumerate(test_iter):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        output = model(data)
        loss = criterion(output, label)
        count += torch.eq(torch.argmax(output, 1), label).sum().item()
    accuracy = count / len(test_iter)
    print('Test accuracy: {:.4f}%'.format(accuracy * 100))


def test_time_adapt(test_iter, model_path, criterion):
    model = torch.load(model_path)
    model.train()
    for _, (data, _) in enumerate(test_iter):
        ...
