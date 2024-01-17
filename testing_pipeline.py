import torch
import higher
import logging

from tqdm import tqdm
from torch.optim import Adam
from algorithm import tent, norm
from utils.average_meter import AverageMeter


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
            count += torch.eq(torch.argmax(output, 1), label).sum().item() / output.shape[0]
        accuracy = count / len(test_iter)
    print('{: <8s}  Test Accuracy: {:.4f}%'.format('(Norm)', accuracy * 100))


def test_with_tent(test_iter, model_path, args):
    model = torch.load(model_path).to(args.BASIC.DEVICE)
    model = tent.configure_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.OPTIM.LEARNING_RATE)
    model = tent.Tent(model, optimizer)

    count = 0
    with torch.no_grad():
        for _, (data, label) in enumerate(test_iter):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            output = model(data)
            count += torch.eq(torch.argmax(output, 1), label).sum().item() / output.shape[0]
        accuracy = count / len(test_iter)
    print('{: <8s}  Test Accuracy: {:.4f}%'.format('(Tent)', accuracy * 100))


def test_with_learned_loss(test_iter, model_path, ll_model_path, args):
    model = torch.load(model_path).to(args.BASIC.DEVICE)
    ll_model = torch.load(ll_model_path).to(args.BASIC.DEVICE)
    params = list(model.parameters()) + list(ll_model.parameters())
    optimizer = Adam(params, lr=args.OPTIM.LEARNING_RATE)
    inner_optimizer = Adam(model.parameters(), lr=args.OPTIM.LEARNING_RATE)
    test_loop = tqdm(enumerate(test_iter), total=len(test_iter))

    count = 0
    model.train_default()
    for _, (data, label) in test_loop:
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        # Meta learning
        with higher.innerloop_ctx(model, inner_optimizer, args.BASIC.DEVICE,
                                  copy_initial_weights=False) as (f_net, diff_opt):
            # Inner loop
            meta_loss_meter = AverageMeter('MetaLossMeter')
            for _ in range(args.inner_epochs):
                spt_logits = f_net(data)
                spt_loss = ll_model(spt_logits)
                diff_opt.step(spt_loss)

                meta_loss_meter.update(spt_loss.item(), args.batch_size)

            logits = f_net(data)
            count += torch.eq(torch.argmax(logits, 1), label).float().mean() / logits.shape[0]

        test_loop.set_description('Adaptive Test')
        test_loop.set_postfix(loss=f'{meta_loss_meter.avg:.4f}')
    accuracy = count / len(test_iter)
    print('(ARM) Test Accuracy: {:.4f}%'.format(accuracy * 100))
