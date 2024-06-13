import torch
import logging
import itertools

from tqdm import tqdm
from torch.optim import Adam
from torch.optim import lr_scheduler
from algorithms.arm import init_algorthm
from datasets.tep_dataset import TEPDataset
from utils.average_meter import AverageMeter
from utils.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader


def train_default(train_iter, eval_iter, model, criterion, args):
    log_tool = logging.getLogger(__name__)
    log_dir = f'{args.log_path}_{args.model.suffix}' \
        if args.model.suffix != '' \
        else f'{args.log_path}'
    writer = SummaryWriter(log_dir=log_dir) if args.write_flag else None
    optimizer = Adam(model.parameters(), lr=args.optim.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.optim.step_size)
    stopping_tool = EarlyStopping(args, save_path=args.model.save_path, verbose=True,
                                  patience=args.util.train.patience, delta=args.util.train.delta)
    global_train_step, global_eval_step = 0, 0
    for epoch in range(args.util.train.epochs):
        model.train()
        train_loop = tqdm(enumerate(train_iter), total=len(train_iter))
        loss_meter = AverageMeter('LossMeter')
        acc_meter = AverageMeter('AccMeter')
        for _, (data, labels) in train_loop:
            loss_meter.reset()
            acc_meter.reset()
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            output = model(data)
            loss = criterion(output, labels)
            loss_meter.update(loss, args.util.train.batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.util.train.batch_size)

            if args.write_flag:
                writer.add_scalar(tag='training loss', scalar_value=loss.item(), global_step=global_train_step)
                writer.add_scalar(tag='training accuracy', scalar_value=accuracy, global_step=global_train_step)
            train_loop.set_description('Train [{}/{}]'.format('{: <2d}'.format(epoch + 1), args.util.train.epochs))
            train_loop.set_postfix(acc='{:.4f}'.format(acc_meter.avg),
                                   loss='{:.4f}'.format(loss_meter.avg))
            global_train_step += 1
        scheduler.step()

        # torch.cuda.empty_cache()

        model.eval()
        eval_loop = tqdm(enumerate(eval_iter), total=len(eval_iter))
        loss_meter.reset()
        acc_meter.reset()
        for _, (data, labels) in eval_loop:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            output = model(data)
            loss = criterion(output, labels)
            loss_meter.update(loss, args.util.train.batch_size)

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.util.train.batch_size)

            if args.write_flag:
                writer.add_scalar(tag='validation loss', scalar_value=loss.item(), global_step=global_eval_step)
                writer.add_scalar(tag='validation accuracy', scalar_value=accuracy, global_step=global_eval_step)
            eval_loop.set_description('Eval  [{}/{}]'.format('{: <2d}'.format(epoch + 1), args.util.train.epochs))
            eval_loop.set_postfix(acc='{:.4f}'.format(acc_meter.avg),
                                  loss='{:.4f}'.format(loss_meter.avg))
            global_eval_step += 1
        if args.log_flag:
            log_tool.info(f'[EPOCH {epoch + 1: <2d}/{args.util.train.epochs}] ACC: {acc_meter.avg * 100:.4f}%')

        stopping_tool(loss_meter.avg, model)
        if stopping_tool.early_stop:
            print('Early Stopping ...')
            break

    if args.write_flag:
        writer.close()


def train_with_arm(domains, model, criterion, args):
    """
    Adaptive Risk Minimization: powered by mata-learning
    """
    # writer = SummaryWriter(log_dir=f'{args.log_dir}core{args.ckpt_suffix}')
    log_tool = logging.getLogger(__name__)
    split_ratio = {'train': args.dataset.split_ratio[0],
                   'eval': args.dataset.split_ratio[1]}
    algorithm = init_algorthm(args, model, criterion, method=args.algorithm.arm)
    stopping_tool = EarlyStopping(args, save_path=args.model.save_path, verbose=True)
    global_train_step, global_eval_step = 0, 0

    train_iters, eval_iters = [], []
    for domain in domains:
        dataset = TEPDataset(args.dataset.path, split_ratio,
                             {'source': domain, 'target': args.domain.target},
                             'train', seed=args.random_seed)
        train_iters.append(DataLoader(dataset, batch_size=args.util.train.batch_size, shuffle=True))
        dataset = TEPDataset(args.PATH.DATA_PATH, split_ratio,
                             {'source': domain, 'target': args.domain.target},
                             'eval', seed=args.random_seed)
        eval_iters.append(DataLoader(dataset, batch_size=args.util.train.batch_size, shuffle=True))

    train_iter = list(itertools.chain(*train_iters))
    eval_iter = list(itertools.chain(*eval_iters))

    for epoch in range(args.util.train.epochs):
        train_loop = tqdm(enumerate(train_iter), total=len(train_iter))
        train_loss_meter = AverageMeter('TrainLossMeter')
        train_acc_meter = AverageMeter('TrainAccMeter')
        for _, (data, labels) in train_loop:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            train_logits, _ = algorithm.learn(data, labels)
            train_loss = criterion(train_logits, labels)
            train_accuracy = torch.eq(torch.argmax(train_logits, 1), labels).float().mean()

            train_loss_meter.update(train_loss, args.util.train.batch_size)
            train_acc_meter.update(train_accuracy, args.util.train.batch_size)
            train_loop.set_description(f'Train [{epoch}/{args.util.train.epochs}]')
            train_loop.set_postfix(loss=f'{train_loss_meter.avg:.4f}',
                                   acc=f'{train_acc_meter.avg:.4f}')

        eval_loss_meter = AverageMeter('EvalLossMeter')
        eval_acc_meter = AverageMeter('EvalAccMeter')
        eval_loop = tqdm(enumerate(eval_iter), total=len(eval_iter))
        for _, (data, labels) in eval_loop:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            eval_logits = algorithm.predict(data)
            eval_loss = criterion(eval_logits, labels)
            eval_accuracy = torch.eq(torch.argmax(eval_logits, 1), labels).float().mean()
            eval_acc_meter.update(eval_accuracy, args.util.train.batch_size)
            eval_loss_meter.update(eval_loss.item(), args.util.train.batch_size)

            eval_loop.set_description(f'Eval [{epoch}/{args.util.train.epochs}]')
            eval_loop.set_postfix(loss=f'{eval_loss_meter.avg:.4f}',
                                  acc=f'{eval_acc_meter.avg:.4f}')
        if args.BASIC.LOG_FLAG:
            log_tool.info(f'[EPOCH {epoch + 1: <2d}/{args.util.train.epochs}] ACC: {eval_acc_meter.avg * 100:.4f}%')
        stopping_tool(eval_loss_meter.avg, algorithm)

        if stopping_tool.early_stop:
            print('Early Stopping ...')
            break

    # writer.close()
