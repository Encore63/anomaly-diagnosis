import torch.jit
import torch.nn as nn

from copy import deepcopy
from algorithms.comp import jmds
from algorithms.comp.sam import SAM
from algorithms.comp import trans_norm


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
        self.divider = deepcopy(model).eval()
        self.model = configure_model(model)
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
            # outputs = forward_and_adapt(x, self.model, self.divider, self.optimizer,
            #                             self.use_entropy, self.weighting)
            outputs = forward_and_adapt(x, self.model, self.divider, self.optimizer)

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


# @torch.enable_grad()  # ensure grads in possible no grad context for testing
# def forward_and_adapt(x, model, divider, optimizer, use_entropy, weighting):
#     """
#     Forward and adapt model on batch of data.
#     `Measure entropy of the model prediction, take gradients, and update params.`
#     """
#     # forward
#     certain_data, uncertain_data, certain_idx, uncertain_idx, weight = domain_division(divider, x,
#                                                                                        use_entropy=use_entropy,
#                                                                                        weighting=weighting)
#     # domain_idx = {'source': certain_idx,
#     #               'target': uncertain_idx}
#     # replace_tn_layer(model, domain_idx)
#     # print(model)
#     # outputs = model(x)
#
#     u_outputs = model(uncertain_data)
#     # model = configure_model(model, True)
#     c_outputs = model(certain_data)
#     if len(c_outputs.shape) == 1:
#         c_outputs = c_outputs.unsqueeze(dim=0)
#     if len(u_outputs.shape) == 1:
#         u_outputs = u_outputs.unsqueeze(dim=0)
#     outputs = domain_merge(c_outputs, u_outputs, certain_idx, uncertain_idx)
#
#     # adapt
#     c_loss = softmax_entropy(c_outputs).mean(0)
#     u_loss = softmax_entropy(u_outputs).mean(0)
#     loss = c_loss * weight[0] + u_loss * weight[1]
#     # loss = softmax_entropy(outputs).mean(0)
#     loss.backward()
#     if isinstance(optimizer, SAM):
#         optimizer.first_step(zero_grad=True)
#         (softmax_entropy(model(uncertain_data)).mean(0) - softmax_entropy(model(certain_data)).mean(0)).backward()
#         optimizer.second_step(zero_grad=True)
#     else:
#         optimizer.step()
#         optimizer.zero_grad()
#
#     return outputs


@torch.enable_grad()
def forward_and_adapt(x, model, divider, optimizer, out_layer='avg_pool'):
    """
    Forward and adapt model on batch of data.
    `Measure entropy of the model prediction, take gradients, and update params.`
    """
    # forward

    weight = jmds.joint_model_data_score(x, divider, out_layer, num_classes=10)
    output = model(x)
    # weight, output = map(torch.autograd.Variable, (weight, output))

    kl_loss = jmds.mix_up(x, weight, output, model)
    ent_loss = softmax_entropy(output).mean(0)
    loss = ent_loss + kl_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output


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


def replace_tn_layer(model, domain_idx):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm1d):
            tn = trans_norm.TransNorm1d(module.num_features, domain_idx,
                                        module.eps, module.momentum,
                                        module.affine, module.track_running_stats).cuda()
            setattr(model, name, tn)  # 使用 setattr 替换 BatchNorm1d
        elif isinstance(module, nn.BatchNorm2d):
            tn = trans_norm.TransNorm2d(module.num_features, domain_idx,
                                        module.eps, module.momentum,
                                        module.affine, module.track_running_stats).cuda()
            setattr(model, name, tn)  # 使用 setattr 替换 BatchNorm2d
        elif isinstance(module, trans_norm.TransNorm1d) or isinstance(module, trans_norm.TransNorm2d):
            module.domain_idx = domain_idx
        else:
            replace_tn_layer(module, domain_idx)  # 递归替换子模块


def configure_model(model, freeze=False):
    """
    Configure model for use with div-tent.
    """
    # train mode, because div-tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statistics
    for n, m in model.named_modules():
        if n == 'fc' or n == 'classifier':
            m.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            if freeze:
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
