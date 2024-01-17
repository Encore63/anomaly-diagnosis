import math
import torch
import higher
import numpy as np

from torch import nn


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
