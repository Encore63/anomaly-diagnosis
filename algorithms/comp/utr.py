import torch
import numpy as np

from copy import deepcopy


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def euclidean_dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def get_mean(f1, f2):
    f1 = f1.view(f1.shape[0], f1.shape[1], -1).cpu().detach().numpy()
    f2 = f2.view(f2.shape[0], f2.shape[1], -1).cpu().detach().numpy()
    # batch size
    x = f1.shape[0]
    # channels
    n = f1.shape[1]

    mean = []
    for nn in range(0, n):
        dis = 0
        for i in range(0, x):
            dis += euclidean_dist(f1[i, nn, :], f2[i, nn, :])
        dis = dis / x
        mean.append(dis)

    mean = torch.from_numpy(np.array(mean))
    return mean


def capture_unc(pre, network, steps=1):
    network.load_state_dict(pre)
    param = network.state_dict()

    for (
            k,
            v,
    ) in param.items():
        for _ in range(steps):
            _a = torch.rand_like(param[k], dtype=torch.float)
            _a = (_a - 0.5) / 10 + 1
            param[k] = torch.mul(param[k], _a)
    network.load_state_dict(param)

    return network


def calc_utr_d(_model, _data):
    pre_param = deepcopy(_model.state_dict())
    pre_output = _model(_data)
    _model = capture_unc(pre_param, _model)
    output = _model(_data)
    utr_d: torch.Tensor = 1 / 4 * get_mean(pre_output, output).cuda()

    # q_utr_d
    utr_d = torch.sigmoid(-utr_d)

    return utr_d


def calc_utr_i(_model, _data):
    pre_param = deepcopy(_model.state_dict())
    pre_output = _model(_data)
    _model = capture_unc(pre_param, _model)
    output = _model(_data)
    sub_output = output - pre_output
    utr_i = torch.sum(sub_output * sub_output, dim=1)

    threshold = min((utr_i.mean()) * 3, utr_i.max())

    return utr_i, threshold


if __name__ == '__main__':
    from datasets.tep_dataset import TEPDataset
    from torch.utils.data.dataloader import DataLoader

    dataset = TEPDataset(src_path='../../data/TEP', transfer_task=[[1], [2]])
    subset = dataset.get_subset('test')
    data_iter = DataLoader(subset, batch_size=128, shuffle=True)
    model = torch.load(r'../../checkpoints/best_model_resnet_1.pth')
    for x, y in data_iter:
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        d, (i, t) = calc_utr_d(model, x), calc_utr_i(model, x)
        print(d, d.shape)
        print(i, i.shape, t)
        break
