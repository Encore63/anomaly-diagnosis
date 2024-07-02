import torch
from torch import nn


class BayesianBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = BayesianBatchNorm(child, prior)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BayesianBatchNorm.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        replace_mods = BayesianBatchNorm.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert 0 <= prior <= 1
        super(BayesianBatchNorm, self).__init__()
        self.layer = layer
        self.layer.eval()
        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=0.1,
                                   track_running_stats=True).cuda()
        self.normed_div_mean = torch.zeros(1).cuda()

    def forward(self, input):
        # if self.norm.training is True:
        self.norm(input)
        self.norm.eval()
        source_distribution = torch.distributions.MultivariateNormal(self.layer.running_mean, (
                self.layer.running_var + 0.00001) * torch.eye(
            self.layer.running_var.shape[0]).cuda())
        target_distribution = torch.distributions.MultivariateNormal(self.norm.running_mean, (
                self.norm.running_var + 0.00001) * torch.eye(
            self.norm.running_var.shape[0]).cuda())

        self.div = (0.5 * torch.distributions.kl_divergence(source_distribution,
                                                            target_distribution) + 0.5 * torch.distributions.kl_divergence(
            target_distribution, source_distribution))

        self.div_values = self.div
        self.prior = self.normed_div_mean

        running_mean = (self.prior * self.layer.running_mean + (1 - self.prior) * self.norm.running_mean)
        running_var = (self.prior * self.layer.running_var) + (1 - self.prior) * self.norm.running_var + self.prior * (
                    1 - self.prior) * ((self.layer.running_mean - self.norm.running_mean) ** (2))

        output = (input - running_mean[None, :, None, None]) / torch.sqrt(
            running_var[None, :, None, None] + self.layer.eps) * self.layer.weight[None, :, None,
                                                                 None] + self.layer.bias[None, :, None, None]

        return output
