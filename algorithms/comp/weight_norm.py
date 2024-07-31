import torch

from torch import nn
from einops import rearrange


def _mahalanobis_distance(x, _mu, _sigma):
    identity = torch.eye(_sigma.size(-1), device=_sigma.device)
    _sigma += 1e-5 * identity
    return (x - _mu) @ torch.inverse(_sigma) @ (x - _mu).T


class WeightNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 use_weight=True, residual=False):
        super(WeightNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.use_weight = use_weight
        self.residual = residual
        self.source_mean = self.running_mean
        self.source_var = self.running_var
        self.register_buffer('weights', torch.ones(num_features))

    def forward(self, x):
        if self.use_weight:
            _var_w = self.get_variant_weight(x)
            if self.residual:
                x = (1 + _var_w) * x
            else:
                x = _var_w * x
        x = super(WeightNorm, self).forward(x)

        return x

    def get_variant_weight(self, x):
        weights = []
        features = rearrange(x, 'b s t c -> (b t c) s')

        p = True
        for var in range(features.shape[-1]):
            index = torch.ones(features.shape[-1])
            index[var] = 0
            index = list(map(bool, index))
            part_features = features[:, index]

            _mean = self.source_mean.view((1, self.num_features))
            _part_mean = _mean[:, index]
            _sigma = ((features - _mean).T @ (features - _mean)) / self.num_features
            _part_sigma = ((part_features - _part_mean).T @ (part_features - _part_mean)) / (self.num_features - 1)
            _distance = _mahalanobis_distance(features, _mean, _sigma)
            _part_distance = _mahalanobis_distance(part_features, _part_mean, _part_sigma)

            var_weight = self.num_features * ((_distance - _part_distance).mean() / _part_distance.mean())
            weights.append(var_weight)

        return torch.Tensor(weights)


if __name__ == '__main__':
    from datasets.tep_dataset import TEPDataset
    from torch.utils.data.dataloader import DataLoader

    subset = TEPDataset(r'../../data/TEP', [[1], [2]]).get_subset('test')
    data_iter = DataLoader(subset, batch_size=16, shuffle=False)
    data = next(enumerate(data_iter))[1][0]
    data = rearrange(data, 'b c t s -> b s t c')

    wn = WeightNorm(50, use_weight=True, residual=True)
    w = wn.get_variant_weight(data)
    print(w, len(w))
