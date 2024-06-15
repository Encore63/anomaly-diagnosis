from torch import nn


class HookManager(object):
    def __init__(self, model: nn.Module):
        self._model = model
        self._hooks = dict()
        self._activations = dict()
        self._modules = dict(self._model.named_modules()).keys()

    def _get_hook(self, layer_name):
        def hook(model, _input, _output):
            self._activations[layer_name] = _output.detach()
        return hook

    def register_hook(self, layer_name):
        layer = dict(self._model.named_modules())[layer_name]
        assert isinstance(layer, nn.Module), "model type error!"
        self._hooks[layer_name] = layer.register_forward_hook(self._get_hook(layer_name))

    def remove_hooks(self):
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()

    def get_activations(self, _input_data):
        self._model(_input_data)
        return self._activations

    def get_modules(self):
        return list(self._modules)
