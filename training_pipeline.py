import pathlib

import torch
import higher

from tqdm import tqdm
from torch.optim import Adam
from torch.optim import lr_scheduler
from datasets.tep_dataset import TEPDataset
from utils.average_meter import AverageMeter
from utils.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader


def train_default(train_iter, eval_iter, model, criterion, args):
    writer = SummaryWriter(log_dir=f'{args.PATH.LOG_PATH}_{args.MODEL.CKPT_SUFFIX}')
    optimizer = Adam(model.parameters(), lr=args.TRAINING.LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.TRAINING.STEP_SIZE)
    stopping_tool = EarlyStopping(args, save_path=args.PATH.CKPT_PATH, verbose=True)
    global_train_step, global_eval_step = 0, 0
    for epoch in range(args.TRAINING.EPOCHS):
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
            loss_meter.update(loss, args.TRAINING.BATCH_SIZE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.TRAINING.BATCH_SIZE)

            writer.add_scalar(tag='training loss', scalar_value=loss.item(), global_step=global_train_step)
            writer.add_scalar(tag='training accuracy', scalar_value=accuracy, global_step=global_train_step)
            train_loop.set_description('Train [{}/{}]'.format('{: <2d}'.format(epoch + 1), args.TRAINING.EPOCHS))
            train_loop.set_postfix(acc='{:.4f}'.format(acc_meter.avg),
                                   loss='{:.4f}'.format(loss_meter.avg))
            global_train_step += 1
        scheduler.step()

        model.eval()
        eval_loop = tqdm(enumerate(eval_iter), total=len(eval_iter))
        loss_meter.reset()
        acc_meter.reset()
        for _, (data, labels) in eval_loop:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            output = model(data)
            loss = criterion(output, labels)
            loss_meter.update(loss, args.TRAINING.BATCH_SIZE)

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.TRAINING.BATCH_SIZE)

            writer.add_scalar(tag='validation loss', scalar_value=loss.item(), global_step=global_eval_step)
            writer.add_scalar(tag='validation accuracy', scalar_value=accuracy, global_step=global_eval_step)
            eval_loop.set_description('Eval  [{}/{}]'.format('{: <2d}'.format(epoch + 1), args.TRAINING.EPOCHS))
            eval_loop.set_postfix(acc='{:.4f}'.format(acc_meter.avg),
                                  loss='{:.4f}'.format(loss_meter.avg))
            global_eval_step += 1

        stopping_tool(loss_meter.avg, model)
        if stopping_tool.early_stop:
            print('Early Stopping ...')
            break

    writer.close()


def train_with_learned_loss(domains, model, ll_model, criterion, args):
    """
    ARM-LL: Adaptive Risk Minimization (Learned Loss)
    """
    # writer = SummaryWriter(log_dir=f'{args.log_dir}_{args.ckpt_suffix}')
    params = list(model.parameters()) + list(ll_model.parameters())
    optimizer = Adam(params, lr=args.TRAINING.LEARNING_RATE)
    inner_optimizer = Adam(model.parameters(), lr=args.TRAINING.LEARNING_RATE)
    stopping_tool = EarlyStopping(args, save_path=args.PATH.CKPT_PATH, verbose=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.TRAINING.STEP_SIZE)
    global_train_step, global_eval_step = 0, 0

    train_iters, eval_iters = [], []
    for domain in domains:
        dataset = TEPDataset(args.PATH.DATA_PATH, args.BASIC.SPLIT_RATIO, {'source': domain, 'target': None},
                             'train', seed=args.BASIC.RANDOM_SEED)
        train_iters.append(DataLoader(dataset, batch_size=args.TRAINING.BATCH_SIZE, shuffle=True))
        dataset = TEPDataset(args.PATH.DATA_PATH, args.BASIC.SPLIT_RATIO, {'source': domain, 'target': None},
                             'eval', seed=args.BASIC.RANDOM_SEED)
        eval_iters.append(DataLoader(dataset, batch_size=args.TRAINING.BATCH_SIZE, shuffle=True))
    for epoch in range(args.TRAINING.EPOCHS):
        for train_iter, eval_iter in zip(train_iters, eval_iters):
            train_loop = tqdm(enumerate(train_iter), total=len(train_iter))
            for _, (data, labels) in train_loop:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                # Meta learning
                with higher.innerloop_ctx(model, inner_optimizer, args.BASIC.DEVICE,
                                          copy_initial_weights=False) as (f_net, diff_opt):
                    # Inner loop
                    meta_loss_meter = AverageMeter('MetaLossMeter')
                    meta_acc_meter = AverageMeter('MetaAccMeter')
                    for _ in range(args.inner_epochs):
                        spt_logits = f_net(data)
                        spt_loss = ll_model(spt_logits)
                        diff_opt.step(spt_loss)

                        meta_accuracy = torch.eq(torch.argmax(spt_logits, 1), labels).float().mean()
                        meta_acc_meter.update(meta_accuracy, args.TRAINING.BATCH_SIZE)
                        meta_loss_meter.update(spt_loss.item(), args.TRAINING.BATCH_SIZE)

                    loss_meter = AverageMeter('LossMeter')
                    acc_meter = AverageMeter('AccMeter')
                    domain_logits = f_net(data)
                    domain_loss = criterion(domain_logits, labels)
                    domain_loss.backward()

                    accuracy = torch.eq(torch.argmax(domain_logits, 1), labels).float().mean()
                    acc_meter.update(accuracy, args.TRAINING.BATCH_SIZE)
                    loss_meter.update(domain_loss.item(), args.TRAINING.BATCH_SIZE)

                optimizer.step()
                optimizer.zero_grad()

                train_loop.set_description(f'Train [{epoch}/{args.TRAINING.EPOCHS}]')
                train_loop.set_postfix(meta_loss=f'{meta_loss_meter.avg:.4f}',
                                       loss=f'{loss_meter.avg:.4f}',
                                       meta_acc=f'{meta_acc_meter.avg:.4f}',
                                       acc=f'{acc_meter.avg:.4f}')

            eval_loss_meter = AverageMeter('LossMeter')
            eval_acc_meter = AverageMeter('EvalAccMeter')
            eval_loop = tqdm(enumerate(eval_iter), total=len(eval_iter))
            for _, (data, labels) in eval_loop:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                eval_logits = f_net(data)
                eval_loss = criterion(eval_logits, labels)
                eval_accuracy = torch.eq(torch.argmax(eval_logits, 1), labels).float().mean()
                eval_acc_meter.update(eval_accuracy, args.TRAINING.BATCH_SIZE)
                eval_loss_meter.update(eval_loss.item(), args.TRAINING.BATCH_SIZE)
                eval_loop.set_description(f'Eval [{epoch}/{args.TRAINING.EPOCHS}]')
                eval_loop.set_postfix(loss=f'{eval_loss_meter.avg:.4f}',
                                      acc=f'{eval_acc_meter.avg:.4f}')
            stopping_tool(eval_loss_meter.avg, model)

        if stopping_tool.early_stop:
            torch.save(ll_model, pathlib.Path(args.PATH.CKPT_PATH).joinpath('learned_loss.pth'))
            print('Early Stopping ...')
            break

    # writer.close()
