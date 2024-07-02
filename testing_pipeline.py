import torch

from torch import nn
from tqdm import tqdm
from utils.average_meter import AverageMeter
from algorithms import tent, norm, delta, divtent


def test_default(test_iter, model_path, args):
    model = torch.load(model_path).to(args.device)
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
    model = torch.load(model_path).to(args.device)
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
    model = torch.load(model_path).to(args.device)
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.optim.learning_rate)
    model = tent.Tent(model, optimizer, steps=args.algorithm.steps)

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
    algorithm = torch.load(model_path).to(args.device)
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
            loss_meter.update(loss, args.util.test.batch_size)
            count += torch.eq(torch.argmax(logits, 1), label).float().mean()

        test_loop.set_description('ARM Test')
        test_loop.set_postfix(loss=f'{loss_meter.avg:.4f}')
    accuracy = count / len(test_iter)
    print('(ARM) Test Accuracy: {:.4f}%'.format(accuracy * 100))


def test_with_data_division(test_iter, model_path, args):
    model = torch.load(model_path).to(args.device)
    model = divtent.configure_model(model)
    params, param_names = divtent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.optim.learning_rate)
    # optimizer = SAM(model.parameters(), base_optimizer=torch.optim.Adam)
    model = divtent.DivTent(model, optimizer, steps=1,
                            use_entropy=args.algorithm.use_entropy,
                            weighting=args.algorithm.weighting)

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
    from utils.data_utils import domain_division

    old_model = torch.load(model_path).to(args.device)
    model = delta.DELTA(args.algorithm, old_model)

    count = 0
    use_division = False
    with torch.no_grad():
        for _, (data, label) in enumerate(test_iter):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
                if use_division:
                    certain_data, uncertain_data, certain_idx, uncertain_idx, weight = \
                        domain_division(old_model, data,
                                        use_entropy=args.algorithm.use_entropy,
                                        weighting=args.algorithm.weighting)
                    model.classifier_adapt(uncertain_data)
                    model.reset()

            output = model(data)
            count += torch.eq(torch.argmax(output, 1), label).float().mean()
        accuracy = count / len(test_iter)
    print('{: <8s}  Test Accuracy: {:.4f}%'.format('(Delta)', accuracy * 100))


if __name__ == '__main__':
    from copy import deepcopy
    from datasets.getter import get_dataset
    from torch.utils.data.dataloader import DataLoader
    from utils.data_utils import domain_division, domain_merge

    model = torch.load(r'./checkpoints/best_model_conv-former_1.pth').to('cuda')
    divider = deepcopy(model)
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params)
    model = tent.Tent(model, optimizer, steps=1)

    count = 0
    tep_dataset = get_dataset(dataset_name='tep', transfer_task=[1, 2], dataset_mode='test')
    data_iter = DataLoader(dataset=tep_dataset, batch_size=128, shuffle=False)
    for x, y in data_iter:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
            c_data, u_data, c_index, u_index, w = domain_division(divider, x, use_entropy=True, weighting=True)
            c_data, u_data = c_data.cuda(), u_data.cuda()
        # c_output = model(c_data)
        u_output = model(u_data)

        count += torch.eq(torch.argmax(u_output, 1), y[u_index]).float().mean()
    print(f'acc: {count / len(data_iter)}')
