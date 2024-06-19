import torch.jit
import torch.nn as nn

from copy import deepcopy
from algorithms.comp.sam import SAM
from utils.loss import conjugate_loss
from algorithms.comp import trans_norm
from utils.data_utils import domain_division, domain_merge


@torch.enable_grad()
def classifier_adapt(x: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer):
    # configure classifier
    model.train()
    model.requires_grad_(False)
    for name, m in model.named_modules():
        if name == 'fc' or name == 'classifier':
            m.requires_grad_(True)

    output = model(x)
    loss = (torch.softmax(output, dim=1) * torch.log_softmax(output, dim=1)).sum(dim=1).mean(0)
    optimizer.zero_grad()
    loss.backward()
    if isinstance(optimizer, SAM):
        optimizer.first_step(zero_grad=True)
        (torch.softmax(output, dim=1) * torch.log_softmax(output, dim=1)).sum(dim=1).mean(0).backward()
        optimizer.second_step(zero_grad=True)
    else:
        optimizer.step()

    return output.detach()


class DivTent(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, use_entropy=False, weighting=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.use_entropy = use_entropy
        self.weighting = weighting

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer,
                                        self.use_entropy, self.weighting)

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


def division_loss(probs):
    msoftmax = probs.mean(dim=0)
    div_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    return div_loss


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, use_entropy, weighting):
    """
    Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    certain_data, uncertain_data, certain_idx, uncertain_idx, weight = domain_division(model, x,
                                                                                       use_entropy=use_entropy,
                                                                                       weighting=weighting)

    # weighted_data = domain_merge(certain_data, uncertain_data, certain_idx, uncertain_idx)
    # outputs = model(weighted_data)

    all_outputs = model(x)
    # model = configure_model(model, True)
    u_outputs = model(uncertain_data)
    c_outputs = model(certain_data)
    # model = configure_model(model, False)
    if len(c_outputs.shape) == 1:
        c_outputs = c_outputs.unsqueeze(dim=0)
    if len(u_outputs.shape) == 1:
        u_outputs = u_outputs.unsqueeze(dim=0)
    outputs = domain_merge(c_outputs, u_outputs, certain_idx, uncertain_idx)

    # adapt
    c_loss = conjugate_loss(c_outputs).mean(0)
    u_loss = conjugate_loss(u_outputs).mean(0)
    loss = c_loss * weight[0] + u_loss * weight[1]
    loss.backward()
    if isinstance(optimizer, SAM):
        optimizer.first_step(zero_grad=True)
        (softmax_entropy(model(uncertain_data)).mean(0) - softmax_entropy(model(certain_data)).mean(0)).backward()
        optimizer.second_step(zero_grad=True)
    else:
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
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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


def find_bn_layer(parent):
    replace_mods = []
    if parent is None:
        return []
    for name, child in parent.named_children():
        if isinstance(child, nn.BatchNorm2d):
            module = trans_norm.TransNorm2d(child.num_features)
            replace_mods.append((parent, name, module))
        elif isinstance(child, nn.BatchNorm1d):
            module = trans_norm.TransNorm1d(child.num_features)
            replace_mods.append((parent, name, module))
        else:
            replace_mods.extend(find_bn_layer(child))
    return replace_mods


def configure_model(model, freeze=False, norm_type='bn'):
    """
    Configure model for use with tent.
    """
    if norm_type == 'tn':
        replace_mods = find_bn_layer(model)
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model
    else:
        # train mode, because tent optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what tent updates
        model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statistics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.requires_grad_(True)
                if freeze:
                    model.eval()
                    m.training = False
                else:
                    m.training = True
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
