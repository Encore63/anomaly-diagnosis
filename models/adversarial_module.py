import torch

from torch import nn
from typing import Any
from torchinfo import summary
from torch.autograd import Function


class ReverseLayer(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x, alpha = kwargs['x'], kwargs['alpha']
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_outputs = torch.tensor(grad_outputs)
        output = torch.neg(grad_outputs) * ctx.alpha
        return output, None


class AdversarialModule(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super(AdversarialModule, self).__init__()
        self.adversarial_block = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.BatchNorm2d(num_features=hidden_dims[0]),
            nn.ReLU()
        )
        for dim in range(1, len(hidden_dims)):
            self.adversarial_block.add_module(name='fc_{}'.format(dim),
                                              module=nn.Linear(hidden_dims[dim - 1], hidden_dims[dim]))
            self.adversarial_block.add_module(name='bn_{}'.format(dim),
                                              module=nn.BatchNorm2d(num_features=hidden_dims[dim]))
            self.adversarial_block.add_module(name='relu_{}'.format(dim),
                                              module=nn.ReLU())

    def forward(self, x, alpha):
        reversed_x = ReverseLayer.apply(x=x, alpha=alpha)
        output = self.adversarial_block(reversed_x)
        return output
