from losses import Loss
from collections import OrderedDict
from .module import Module
import torch


class Sequential(Module):

    def __init__(self, modules=None, loss_func=None):
        super(Sequential, self).__init__(trainable=False)
        self._modules = OrderedDict()

        if modules is not None:
            if isinstance(modules, OrderedDict):
                for name, module in modules.items():
                    self.add_module(module, name)

            elif isinstance(modules, list):
                for idx, module in enumerate(modules):
                    name = getattr(module, 'name')
                    mod_name = name if name is not None else str(idx)
                    self.add_module(module, mod_name)
            else:
                raise TypeError('Given parameter {} is not valid '.format(type(modules)))

        self._loss = loss_func

    def add_module(self, module: Module, name):
        print("Added Module Name: {} ".format(name))
        if module is None or isinstance(module, Loss):
            raise ValueError('Given object type is Loss is not valid !')
        if module is None or not isinstance(module, Module):
            raise ValueError('Given object type {} is not Module '.format(type(module)))
        if name is None or len(name) == 0:
            raise ValueError('Given name {} is not valid'.format(name))

        self._modules[name] = module
        act_layer = getattr(module, 'activation', None)

        if act_layer is not None:
            self.add_module(act_layer, name + "_" + act_layer.__class__.__name__)

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor=False):

        out = input

        for module in self._modules.values():
            out = module.forward(out)

        loss_val = None if target is None else self._loss.forward(out, target)
        return out, loss_val

    def backward(self):

        gradwrtoutputt = self._loss.backward()
        for module in reversed(list(self._modules.values())):
            gradwrtoutputt = module.backward(gradwrtoutputt)

    def predict(self, input):
        output, _ = self.forward(input)
        _, y_pred = output.max(1)

        return y_pred

    def accuracy(self, input, target):
        y_pred = self.predict(input)
        return (y_pred == target).type(torch.FloatTensor).mean(), y_pred

    def print_model(self):
        for name, module in self._modules.items():
            print("Name: {}".format(name))

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss_func):
        self._loss = loss_func

    @property
    def trainable_modules(self):
        for module in self._modules.values():
            if module.trainable:
                yield module