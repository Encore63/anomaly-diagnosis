import torch
import pathlib
import argparse

from tqdm import tqdm
from models.tenet import TENet
from torchvision import transforms
from torch.utils.data import dataloader
from torch.optim.lr_scheduler import StepLR
from datasets.tep_dataset import TEPDataset
from utils.average_meter import AverageMeter
from torch.utils.tensorboard import SummaryWriter


def train(train_iter, eval_iter, _model, _optimizer, _scheduler, _criterion, epochs, writer, save_path):
    for epoch in range(epochs):
        _model.train()
        train_loop = tqdm(enumerate(train_iter), total=len(train_iter))
        loss_meter = AverageMeter('LossMeter')
        acc_meter = AverageMeter('AccMeter')
        for i, (images, labels) in train_loop:
            loss_meter.reset()
            acc_meter.reset()
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            output = _model(images)
            loss = criterion(output, labels)
            loss_meter.update(loss, args.batch_size)

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.batch_size)

            train_loop.set_description('Train [{}/{}]'.format('{: <2d}'.format(epoch + 1), epochs))
            train_loop.set_postfix(acc='{:.4f}'.format(acc_meter.avg),
                                   loss='{:.4f}'.format(loss_meter.avg))
        # _scheduler.step()

        _model.eval()
        eval_loop = tqdm(enumerate(eval_iter), total=len(eval_iter))
        loss_meter.reset()
        acc_meter.reset()
        for i, (images, labels) in eval_loop:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            output = _model(images)
            loss = criterion(output, labels)
            loss_meter.update(loss, args.batch_size)

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.batch_size)

            eval_loop.set_description('Eval  [{}/{}]'.format('{: <2d}'.format(epoch + 1), epochs))
            eval_loop.set_postfix(acc='{:.4f}'.format(acc_meter.avg),
                                  loss='{:.4f}'.format(loss_meter.avg))

        torch.save(_model, save_path.joinpath('tenet_{}'.format(epoch)))


def test(test_iter, model_path, _criterion):
    _model = torch.load(model_path)
    _model.eval()
    count, accuracy = 0, 0
    for i, (image, label) in enumerate(test_iter):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        output = _model(image)
        loss = criterion(output, label)
        count += torch.eq(torch.argmax(output, 1), label).sum().item()
    accuracy = count / len(test_iter)
    print('Test accuracy: {:.4f}%'.format(accuracy * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--data-path', type=str, default=r'./data/TEP')
    parser.add_argument('--split-ratio', type=tuple, default=(0.7, 0.3))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--step-size', type=int, default=5, help='step size of learning rate scheduler')
    parser.add_argument('--s', type=int, default=0, help='source domain')
    parser.add_argument('--t', type=int, default=1, help='target domain')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--log-dir', type=str, default=r'./logs', help='save path of logs')
    parser.add_argument('--output-dir', type=str, default=r'./checkpoints', help='save path of model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = TENet(f1=64, f2=128, depth=8, num_classes=args.num_classes).to(args.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss()

    args.step_size = args.epochs / 10
    lr_scheduler = StepLR(optimizer, step_size=args.step_size)

    log_writer = SummaryWriter(log_dir=args.log_dir)

    args.split_ratio = {'train': args.split_ratio[0],
                        'eval': args.split_ratio[1]}
    data_domains = {'source': args.s, 'target': args.t}
    train_dataset = TEPDataset(args.data_path, args.split_ratio, data_domains, 'train')
    eval_dataset = TEPDataset(args.data_path, args.split_ratio, data_domains, 'eval')
    test_dataset = TEPDataset(args.data_path, args.split_ratio, data_domains, 'test')

    train_dataloader = dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = dataloader.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = dataloader.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train(train_iter=train_dataloader,
          eval_iter=eval_dataloader,
          _model=model,
          _criterion=criterion,
          _optimizer=optimizer,
          _scheduler=lr_scheduler,
          epochs=args.epochs,
          writer=log_writer,
          save_path=pathlib.Path(args.output_dir))

    test(test_iter=test_dataloader,
         model_path=pathlib.Path(args.output_dir).joinpath('tenet.pth'),
         _criterion=criterion)
