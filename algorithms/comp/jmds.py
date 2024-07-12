import math
import torch
import numpy as np

from torch import nn
from utils.hook_manager import HookManager


def gmm(all_fea, pi, mu, all_output):
    Cov = []
    dist = []
    log_probs = []
    epsilon = 1e-6

    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:, i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + epsilon * torch.eye(
            temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += epsilon * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5 * (Covi.shape[0] * np.log(2 * math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)

    return zz, gamma


def KLLoss(input_, target_, coeff):
    epsilon = 1e-6
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (-target_ * torch.log(softmax + epsilon)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)


def mix_up(_x, c_batch, t_batch, model, class_num=10, alpha=1.0):
    # weight mix-up
    if alpha == 0:
        outputs = model(_x)
        return KLLoss(outputs, t_batch, c_batch)
    lam = (torch.from_numpy(np.random.beta(alpha, alpha, [len(_x)]))).float().cuda()
    t_batch = torch.eye(class_num).cuda()[t_batch.argmax(dim=1)].cuda()
    shuffle_idx = torch.randperm(len(_x))
    mixed_x = (lam * _x.permute(1, 2, 3, 0) + (1 - lam) * _x[shuffle_idx].permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
    mixed_c = lam * c_batch + (1 - lam) * c_batch[shuffle_idx]
    mixed_t = (lam * t_batch.permute(1, 0) + (1 - lam) * t_batch[shuffle_idx].permute(1, 0)).permute(1, 0)
    mixed_x, mixed_c, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_c, mixed_t))
    mixed_outputs = model(mixed_x)
    return KLLoss(mixed_outputs, mixed_t, mixed_c)


def joint_model_data_score(data, model, layer_name, num_classes):
    hook_tool = HookManager(model)
    hook_tool.register_hook(layer_name)
    all_feas = hook_tool.get_activations(data)[layer_name]
    if len(all_feas.shape) > 2:
        all_feas = all_feas.view(all_feas.shape[0], -1)

    uniform = torch.ones(len(all_feas), num_classes) / num_classes
    uniform = uniform.cuda()

    model.eval()
    all_outputs = model(data)
    all_outputs = torch.softmax(all_outputs, dim=1)

    pi = all_outputs.sum(dim=0)
    mu = all_outputs.T @ all_feas
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm(all_feas, pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)

    for _ in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), all_feas)
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm(all_feas, pi, mu, gamma)
        pred_label = gamma.argmax(dim=1)

    sort_zz = zz.sort(dim=1, descending=True)[0]
    zz_sub = sort_zz[:, 0] - sort_zz[:, 1]

    lpg = zz_sub / zz_sub.max()
    mppl = all_outputs.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    weights = lpg * mppl

    return weights, pred_label.to(torch.long)


if __name__ == '__main__':
    from datasets.tep_dataset import TEPDataset
    from torch.utils.data.dataloader import DataLoader
    tep_dataset = TEPDataset(r'../../data/TEP', transfer_task=[[1], [2]]).get_subset('test')
    data_iter = DataLoader(tep_dataset, batch_size=16, shuffle=True)
    pretrained_model = torch.load(r'../../checkpoints/best_model_resnet_1.pth').cuda()
    for x, y in data_iter:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        pred = pretrained_model(x)
        w = joint_model_data_score(x, pretrained_model, 'avg_pool', 10)
        loss = mix_up(x, w, pred, pretrained_model)
        print(w, w.sum(), loss, sep='\n')
        break
