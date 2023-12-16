import torch

from tqdm import tqdm
from torch.optim import Adam
from torch.optim import lr_scheduler
from utils.average_meter import AverageMeter
from utils.early_stopping import EarlyStopping


def train(train_iter, eval_iter, model, criterion, writer, save_path, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size)
    stopping_tool = EarlyStopping(save_path=save_path, s=args.s, t=args.t)
    global_train_step, global_eval_step = 0, 0
    for epoch in range(args.epochs):
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
            loss_meter.update(loss, args.batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.batch_size)

            writer.add_scalar(tag='training loss', scalar_value=loss.item(), global_step=global_train_step)
            writer.add_scalar(tag='training accuracy', scalar_value=accuracy, global_step=global_train_step)
            train_loop.set_description('Train [{}/{}]'.format('{: <2d}'.format(epoch + 1), args.epochs))
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
            loss_meter.update(loss, args.batch_size)

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.batch_size)

            writer.add_scalar(tag='testing loss', scalar_value=loss.item(), global_step=global_eval_step)
            writer.add_scalar(tag='testing accuracy', scalar_value=accuracy, global_step=global_eval_step)
            eval_loop.set_description('Eval  [{}/{}]'.format('{: <2d}'.format(epoch + 1), args.epochs))
            eval_loop.set_postfix(acc='{:.4f}'.format(acc_meter.avg),
                                  loss='{:.4f}'.format(loss_meter.avg))
            global_eval_step += 1

        stopping_tool(loss_meter.avg, model)
        if stopping_tool.early_stop:
            print('Early Stopping ...')
            break
