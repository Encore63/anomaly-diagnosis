import torch

from tqdm import tqdm
from torch.optim import Adam
from models.tenet import TENet
from utils.data_utils import DataTransform


def test(test_iter, model_path, criterion, args):
    model = torch.load(model_path).to(args.device)
    model.eval()
    count = 0
    for _, (data, label) in enumerate(test_iter):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        output = model(data)
        count += torch.eq(torch.argmax(output, 1), label).sum().item()
    accuracy = count / len(test_iter)
    print('Test Accuracy: {:.4f}%'.format(accuracy * 100))


def adaptive_test(test_iter, model_path, criterion, args):
    """
    TTBA: Test-time Batch-normalization Adaptation
    """
    model = torch.load(model_path).to(args.device)
    assert isinstance(model, TENet), "Invalid model type!"
    model.ext_block_1.requires_grad_(False)
    model.ext_block_1.bn_1.requires_grad_(True)
    model.ext_block_1.bn_1.reset_parameters()
    model.ext_block_1.bn_2.requires_grad_(True)
    model.ext_block_1.bn_2.reset_parameters()

    model.ext_block_2.requires_grad_(False)
    model.ext_block_2.bn_3.requires_grad_(True)
    model.ext_block_2.bn_3.reset_parameters()

    model.class_classifier.requires_grad_(False)

    optimizer = Adam(model.parameters(), lr=args.lr)
    model.train()
    ada_loop = tqdm(enumerate(test_iter), total=len(test_iter))
    for _, (data, label) in ada_loop:
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ada_loop.set_description(f'Test-time Adapt')
        ada_loop.set_postfix(loss=f'{loss: .4f}')

    count = 0
    model.eval()
    for _, (data, label) in enumerate(test_iter):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        output = model(data)
        count += torch.eq(torch.argmax(output, 1), label).sum().item()
    accuracy = count / len(test_iter)
    print('AdaTest Accuracy: {:.4f}%'.format(accuracy * 100))
