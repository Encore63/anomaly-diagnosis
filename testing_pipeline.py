import torch
import higher

from tqdm import tqdm
from torch.optim import Adam
from models.tenet import TENet, ReTENet
from utils.average_meter import AverageMeter
from torch.utils.data.dataloader import DataLoader


def test(test_iter, model_path, args):
    model = torch.load(model_path).to(args.device)
    if args.ttba:
        model.train()
    else:
        model.eval()
    count = 0
    for _, (data, label) in enumerate(test_iter):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        output = model(data)
        count += torch.eq(torch.argmax(output, 1), label).sum().item()
    accuracy = count / len(test_iter)
    print('Test Accuracy: {:.4f}%'.format(accuracy * 100))


def adaptive_test(test_dataset, model_path, criterion, args):
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    model = torch.load(model_path).to(args.device)
    assert isinstance(model, TENet) or isinstance(model, ReTENet), "Invalid model type!"
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
    loss_meter = AverageMeter('LossMeter')
    ada_loop = tqdm(enumerate(test_iter), total=len(test_iter))
    for _, (data, label) in ada_loop:
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        output = model(data)
        loss = criterion(output, label)
        loss_meter.update(loss, args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ada_loop.set_description(f'Test-time Adapt')
        ada_loop.set_postfix(loss=f'{loss_meter.avg: .4f}')

    count = 0
    model.eval()
    test_iter = DataLoader(test_dataset, shuffle=False)
    for _, (data, label) in enumerate(test_iter):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        output = model(data)
        count += torch.eq(torch.argmax(output, 1), label).sum().item()
    accuracy = count / len(test_iter)
    print('AdaTest Accuracy: {:.4f}%'.format(accuracy * 100))


def test_with_learned_loss(test_iter, model_path, ll_model_path, args):
    model = torch.load(model_path).to(args.device)
    ll_model = torch.load(ll_model_path).to(args.device)
    params = list(model.parameters()) + list(ll_model.parameters())
    optimizer = Adam(params, lr=args.lr)
    inner_optimizer = Adam(model.parameters(), lr=args.lr)
    test_loop = tqdm(enumerate(test_iter), total=len(test_iter))

    count = 0
    model.train()
    for _, (data, label) in test_loop:
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        # Meta learning
        with higher.innerloop_ctx(model, inner_optimizer, args.device,
                                  copy_initial_weights=False) as (f_net, diff_opt):
            # Inner loop
            meta_loss_meter = AverageMeter('MetaLossMeter')
            for _ in range(args.inner_epochs):
                spt_logits = f_net(data)
                spt_loss = ll_model(spt_logits)
                diff_opt.step(spt_loss)

                meta_loss_meter.update(spt_loss.item(), args.batch_size)

            logits = f_net(data)
            count += torch.eq(torch.argmax(logits, 1), label).float().mean()

        test_loop.set_description('Adaptive Test')
        test_loop.set_postfix(loss=f'{meta_loss_meter.avg:.4f}')
    accuracy = count / len(test_iter)
    print('Test_LL Accuracy: {:.4f}%'.format(accuracy * 100))
