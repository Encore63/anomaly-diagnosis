import torch
import higher
import logging

from torch import nn
from tqdm import tqdm
from utils.average_meter import AverageMeter
from algorithms import tent, norm, arm, delta, divtent


def test_default(test_iter, model_path, args):
    model = torch.load(model_path).to(args.BASIC.DEVICE)
    model.eval()
    count = 0
    with torch.no_grad():
        for _, (data, label) in enumerate(test_iter):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            output = model(data)
            count += torch.eq(torch.argmax(output, 1), label).sum().item() / output.shape[0]
        accuracy = count / len(test_iter)
    print('{: <9s} Test Accuracy: {:.4f}%'.format('(Default)', accuracy * 100))


def test_with_adaptive_norm(test_iter, model_path, args):
    model = torch.load(model_path).to(args.BASIC.DEVICE)
    model = norm.Norm(model)
    count = 0
    with torch.no_grad():
        for _, (data, label) in enumerate(test_iter):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            output = model(data)
            count += torch.eq(torch.argmax(output, 1), label).float().mean()
        accuracy = count / len(test_iter)
    print('{: <8s}  Test Accuracy: {:.4f}%'.format('(Norm)', accuracy * 100))


def test_with_tent(test_iter, model_path, args):
    model = torch.load(model_path).to(args.BASIC.DEVICE)
    model = tent.configure_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.OPTIM.LEARNING_RATE)
    model = tent.Tent(model, optimizer)

    count = 0
    for _, (data, label) in enumerate(test_iter):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        output = model(data)
        count += torch.eq(torch.argmax(output, 1), label).float().mean()
    accuracy = count / len(test_iter)
    print('{: <8s}  Test Accuracy: {:.4f}%'.format('(Tent)', accuracy * 100))


def test_with_arm(test_iter, model_path, args):
    criterion = nn.CrossEntropyLoss()
    algorithm = torch.load(model_path).to(args.BASIC.DEVICE)
    test_loop = tqdm(enumerate(test_iter), total=len(test_iter))

    count = 0
    algorithm.eval()
    loss_meter = AverageMeter('MetaLossMeter')
    for _, (data, label) in test_loop:
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()

        logits = algorithm.predict(data)
        with torch.no_grad():
            loss = criterion(logits, label)
            loss_meter.update(loss, args.TESTING.BATCH_SIZE)
            count += torch.eq(torch.argmax(logits, 1), label).float().mean()

        test_loop.set_description('ARM Test')
        test_loop.set_postfix(loss=f'{loss_meter.avg:.4f}')
    accuracy = count / len(test_iter)
    print('(ARM) Test Accuracy: {:.4f}%'.format(accuracy * 100))


def test_with_data_division(test_iter, model_path, args):
    model = torch.load(model_path).to(args.BASIC.DEVICE)
    model = divtent.configure_model(model, weight=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.OPTIM.LEARNING_RATE)
    model = divtent.DivTent(model, optimizer, steps=5, use_entropy=args.TESTING.USE_ENTROPY)

    count = 0
    with torch.no_grad():
        for _, (data, label) in enumerate(test_iter):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            output = model(data)
            count += torch.eq(torch.argmax(output, 1), label).float().mean()
        accuracy = count / len(test_iter)
    print('{: <8s}  Test Accuracy: {:.4f}%'.format('(DivTent)', accuracy * 100))


def test_with_delta(test_iter, model_path, args):
    from omegaconf import OmegaConf
    from easydict import EasyDict

    delta_cfg = OmegaConf.load(r'./configs/delta_cfg.yaml')
    model = torch.load(model_path).to(args.BASIC.DEVICE)
    delta_cfg = EasyDict(delta_cfg)
    model = delta.DELTA(delta_cfg, model)

    count = 0
    with torch.no_grad():
        for _, (data, label) in enumerate(test_iter):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            output = model(data)
            count += torch.eq(torch.argmax(output, 1), label).float().mean()
        accuracy = count / len(test_iter)
    print('{: <8s}  Test Accuracy: {:.4f}%'.format('(Delta)', accuracy * 100))
