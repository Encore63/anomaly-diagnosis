import math
import torch
import higher
import numpy as np

from torch import nn
from default import cfg
from typing import Union
from models.mlp import MLP
from models.resnet import resnet18
from models.ctxnet import ContextNet

from datasets.tep_dataset import TEPDataset
from torch.utils.data.dataloader import DataLoader


class ERM(nn.Module):
    def __init__(self, model, loss_fn, device, hparams, init_optim=True, **kwargs):
        super().__init__()
        self.optimizer = None
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

        self.optimizer_name = hparams['optimizer']
        self.learning_rate = hparams['learning_rate']
        self.weight_decay = hparams['weight_decay']

        if init_optim:
            params = self.model.parameters()
            self.init_optimizers(params)

    def init_optimizers(self, params):
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.learning_rate,
                                              weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, params),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay)

    def predict(self, x):
        return self.model(x)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def get_acc(self, logits, labels):
        # Evaluate
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(predictions == labels.detach().cpu().numpy().reshape(-1))
        return accuracy

    def learn(self, images, labels, group_ids=None):

        self.train()

        # Forward
        logits = self.predict(images)
        loss = self.loss_fn(logits, labels)

        self.update(loss)

        stats = {
            'objective': loss.detach().item(),
        }

        return logits, stats


class ARM_BN(ERM):
    def __init__(self, model, loss_fn, device, hparams={}):
        super().__init__(model, loss_fn, device, hparams)

        self.support_size = hparams['support_size']

    def predict(self, x):
        self.model.train()

        n_domains = math.ceil(len(x) / self.support_size)

        logits = []
        for domain_id in range(n_domains):
            start = domain_id * self.support_size
            end = start + self.support_size
            end = min(len(x), end)  # in case final domain has fewer than support size samples
            domain_x = x[start:end]
            domain_logits = self.model(domain_x)
            logits.append(domain_logits)

        logits = torch.cat(logits)

        return logits


class ARM_CML(ERM):
    def __init__(self, model, loss_fn, device, context_net, hparams={}):
        super().__init__(model, loss_fn, device, hparams)

        self.context_net = context_net
        self.support_size = hparams['support_size']
        self.n_context_channels = hparams['n_context_channels']
        self.adapt_bn = hparams['adapt_bn']

        params = list(self.model.parameters()) + list(self.context_net.parameters())
        self.init_optimizers(params)

    def predict(self, x):
        batch_size, c, h, w = x.shape

        if batch_size % self.support_size == 0:
            meta_batch_size, support_size = batch_size // self.support_size, self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size

        if self.adapt_bn:
            out = []
            for i in range(meta_batch_size):
                x_i = x[i * support_size:(i + 1) * support_size]
                context_i = self.context_net(x_i)
                context_i = context_i.mean(dim=0).expand(support_size, -1, -1, -1)
                x_i = torch.cat([x_i, context_i], dim=1)
                out.append(self.model(x_i))
            return torch.cat(out)
        else:
            context = self.context_net(x)  # Shape: batch_size, channels, H, W
            context = context.reshape((meta_batch_size, support_size, self.n_context_channels, h, w))
            context = context.mean(dim=1)  # Shape: meta_batch_size, self.n_context_channels
            context = torch.repeat_interleave(context, repeats=support_size,
                                              dim=0)  # meta_batch_size * support_size, context_size
            x = torch.cat([x, context], dim=1)
            return self.model(x)


class ARM_LL(ERM):
    def __init__(self, model, loss_fn, device, learned_loss_net, hparams=None):
        super().__init__(model, loss_fn, device, hparams)

        if hparams is None:
            hparams = {}
        self.support_size = hparams['support_size']
        self.learned_loss_net = learned_loss_net
        self.n_inner_iter = 1
        self.inner_lr = 1e-1
        self.inner_opt = torch.optim.SGD(model.parameters(),
                                         lr=self.inner_lr)

        params = list(self.model.parameters()) + list(self.learned_loss_net.parameters())
        self.init_optimizers(params)

    def predict(self, x, labels=None, backprop_loss=False):

        self.train()  # see this thread for why this is done https://github.com/facebookresearch/higher/issues/55

        n_domains = math.ceil(len(x) / self.support_size)

        logits = []
        loss = []
        for domain_id in range(n_domains):
            start = domain_id * self.support_size
            end = start + self.support_size
            end = min(len(x), end)  # in case final domain has fewer than support size samples

            domain_x = x[start:end]

            with higher.innerloop_ctx(
                    self.model, self.inner_opt, copy_initial_weights=False) as (f_net, diff_opt):

                # Inner loop
                for _ in range(self.n_inner_iter):
                    spt_logits = f_net(domain_x)
                    spt_loss = self.learned_loss_net(spt_logits)
                    diff_opt.step(spt_loss)

                # Evaluate
                domain_logits = f_net(domain_x)
                logits.append(domain_logits)

                if backprop_loss and labels is not None:
                    domain_labels = labels[start:end]
                    domain_loss = self.loss_fn(domain_logits, domain_labels)
                    domain_loss.backward()
                    loss.append(domain_loss.to('cpu').detach().item())

        logits = torch.cat(logits)

        if backprop_loss:
            return logits, np.mean(loss)
        else:
            return logits

    def learn(self, x, labels, group_ids=None):
        self.train()
        logits, loss = self.predict(x, labels, backprop_loss=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        stats = {}

        return logits, stats


def init_algorthm(model: nn.Module, criterion, method: str = 'ARM_LL'):
    hparams = {'optimizer': 'Adam',
               'learning_rate': cfg.OPTIM.LEARNING_RATE,
               'weight_decay': cfg.OPTIM.WEIGHT_DECAY}
    hparams.setdefault('support_size', cfg.TRAINING.BATCH_SIZE)

    if method == 'ARM_BN':
        algorithm = ARM_BN(model, loss_fn=criterion, device=cfg.BASIC.DEVICE,
                           hparams=hparams).to(cfg.BASIC.DEVICE)
    elif method == 'ARM_LL':
        ll_model = MLP(in_features=cfg.MODEL.NUM_CLASSES, hidden_dim=32,
                       out_features=1, norm_reduce=True).to(cfg.BASIC.DEVICE)
        algorithm = ARM_LL(model=model, loss_fn=criterion, device=cfg.BASIC.DEVICE,
                           learned_loss_net=ll_model, hparams=hparams).to(cfg.BASIC.DEVICE)
    elif method == 'ARM_CML':
        hparams['n_context_channels'] = cfg.MODEL.CONTEXT_CHANNELS
        hparams['adapt_bn'] = cfg.MODEL.ADAPT_BN
        context_net = ContextNet(in_channels=1, out_channels=1, hidden_dim=64, kernel_size=5)
        algorithm = ARM_CML(model, loss_fn=criterion, device=cfg.BASIC.DEVICE, context_net=context_net,
                            hparams=hparams).to(cfg.BASIC.DEVICE)
    else:
        algorithm = ERM(model=model, loss_fn=criterion, device=cfg.BASIC.DEVICE,
                        hparams=hparams).to(cfg.BASIC.DEVICE)

    return algorithm


if __name__ == '__main__':
    data_domains = {'source': int(cfg.DATA.SOURCE), 'target': int(cfg.DATA.TARGET)}
    split_ratio = {'train': cfg.DATA.SPLIT_RATIO[0],
                   'eval': cfg.DATA.SPLIT_RATIO[1]}
    dataset = TEPDataset(r'../data/TEP', split_ratio, data_domains,
                         'test', seed=cfg.BASIC.RANDOM_SEED,
                         time_win=cfg.DATA.TIME_WINDOW)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    algo = init_algorthm(resnet18(in_channels=2), nn.CrossEntropyLoss(), method='ARM_CML')

    for _, (X, y) in enumerate(dataloader):
        X, y = X.cuda(), y.cuda()
        print(algo.learn(X, y))
        break
