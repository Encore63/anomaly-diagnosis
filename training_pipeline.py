import torch

from tqdm import tqdm
from utils.average_meter import AverageMeter


def train(train_iter, eval_iter, _model, _optimizer, _scheduler, _criterion, epochs, writer, save_path, args):
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
            loss = _criterion(output, labels)
            loss_meter.update(loss, args.batch_size)

            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.batch_size)

            writer.add_scalar(tag='training loss', scalar_value=loss_meter.val, global_step=i)
            writer.add_scalar(tag='training accuracy', scalar_value=acc_meter.val, global_step=i)
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
            loss = _criterion(output, labels)
            loss_meter.update(loss, args.batch_size)

            accuracy = torch.eq(torch.argmax(output, 1), labels).float().mean()
            acc_meter.update(accuracy, args.batch_size)

            writer.add_scalar(tag='testing loss', scalar_value=loss_meter.val, global_step=i)
            writer.add_scalar(tag='testing accuracy', scalar_value=acc_meter.val, global_step=i)
            eval_loop.set_description('Eval  [{}/{}]'.format('{: <2d}'.format(epoch + 1), epochs))
            eval_loop.set_postfix(acc='{:.4f}'.format(acc_meter.avg),
                                  loss='{:.4f}'.format(loss_meter.avg))

        torch.save(_model, save_path.joinpath('tenet_{}'.format(epoch)))
