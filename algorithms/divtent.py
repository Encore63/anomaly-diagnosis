import torch.jit
import torch.nn as nn

from copy import deepcopy
from utils.loss import tsallis_entropy
from utils.data_utils import domain_division, domain_merge


class WeightedBatchNorm2d(nn.Module):
    def __init__(self, num_features, weight=None, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(WeightedBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        if weight is not None:
            # assert weight.shape[0] == num_features
            self.weight = nn.Parameter(weight)
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        if self.weight is not None:
            x = self.weight * x
        x = self.bn(x)
        return x


class DivTent(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, use_entropy=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.use_entropy = use_entropy

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer, self.use_entropy)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """
    Entropy of softmax distribution from logits.
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, use_entropy):
    """
    Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    certain_data, uncertain_data, certain_idx, uncertain_idx = domain_division(model, x, use_entropy=use_entropy,
                                                                               weighting=True)
    # delta = 0.9
    certain_data = certain_data
    uncertain_data = uncertain_data
    # data = domain_merge(certain_data, uncertain_data, certain_idx, uncertain_idx)
    c_outputs = model(certain_data)
    u_outputs = model(uncertain_data)
    outputs = domain_merge(c_outputs, u_outputs, certain_idx, uncertain_idx)

    # adapt
    u_loss = softmax_entropy(u_outputs).mean(0)
    c_loss = -softmax_entropy(c_outputs).mean(0)
    loss = u_loss + c_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return outputs


def collect_params(model):
    """
    Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """
    Copy the model and optimizer states for resetting after adaptation.
    """
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """
    Restore the model and optimizer states from copies.
    """
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, weight: torch.Tensor):
    """
    Configure model for use with tent.
    """
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statistics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # m = WeightedBatchNorm2d(m.num_features, weight, m.eps, m.momentum, m.affine, m.track_running_stats)
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
