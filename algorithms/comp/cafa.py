import torch
import numpy as np

from utils.hook_manager import HookManager


def get_prior_classed_statistics(model_path, device, source_iter, num_classes=10):
    _model = torch.load(model_path).to(device)
    model_hook = HookManager(_model)
    model_hook.register_hook('avg_pool')
    classed_features = {cls: [] for cls in range(num_classes)}

    for data, labels in source_iter:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        features = model_hook.get_activations(data)['avg_pool']
        features = features.view(features.shape[0], -1)

        for feature, label in zip(features, labels):
            classed_features[label.item()].append(feature)

    c_features = {cls: torch.stack(features) for cls, features in classed_features.items()}
    del classed_features
    model_hook.remove_hooks()

    c_mu = {cls: features.mean(0) for cls, features in c_features.items()}
    c_sigma = {cls: ((features - c_mu[cls]).T @ (features - c_mu[cls])) / len(features)
               for cls, features in c_features.items()}
    return c_mu, c_sigma


def _calc_statistics(features):
    _mu = features.mean(0)
    _sigma = ((features - _mu).T @ (features - _mu)) / len(features)
    return _mu, _sigma


def _mahalanobis_distance(x, _mu, _sigma):
    identity = torch.eye(_sigma.size(-1), device=_sigma.device)
    _sigma += 1e-5 * identity
    return (x - _mu).T @ torch.inverse(_sigma) @ (x - _mu)


def class_aware_feature_alignment_loss(x, _model, c_mu, c_sigma):
    _y = torch.argmax(torch.softmax(_model(x), 1), 1)
    model_hook = HookManager(_model)
    model_hook.register_hook('avg_pool')
    features = model_hook.get_activations(x)['avg_pool']
    features = features.view(features.shape[0], -1)

    intra_cls_dists = []
    inter_cls_dists = []
    for fea, y in zip(features, _y):
        inter_cls_dist = 0
        intra_cls_dists.append(_mahalanobis_distance(fea, c_mu[y.item()], c_sigma[y.item()]))
        for cls in c_mu.keys():
            inter_cls_dist += _mahalanobis_distance(fea, c_mu[cls], c_sigma[cls])
        inter_cls_dists.append(torch.Tensor(inter_cls_dist))

    model_hook.remove_hooks()
    _intra = torch.stack(intra_cls_dists)
    _inter = torch.stack(inter_cls_dists)
    return torch.log(_intra / _inter).mean(0)


if __name__ == '__main__':
    import time
    from datasets.tep_dataset import TEPDataset
    from torch.utils.data.dataloader import DataLoader

    subsets = TEPDataset(r'../../data/TEP', [[1], [2]]).get_subset()
    data_iter = DataLoader(subsets['train'], batch_size=128, shuffle=False)
    model = torch.load('../../checkpoints/best_model_resnet_1.pth')
    mu, sigma = get_prior_classed_statistics(r'../../checkpoints/best_model_resnet_1.pth', 'cuda', data_iter)
    print(len(mu), len(sigma))

    loss = class_aware_feature_alignment_loss(torch.randn((128, 1, 10, 50)).cuda(), model, mu, sigma)
    print(loss)
