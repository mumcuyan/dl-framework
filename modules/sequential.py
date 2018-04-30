from .dropout import Dropout
from .losses import Loss
from collections import OrderedDict
from .activations import *
import torch


class Sequential(Module):

    def __init__(self, modules=None, loss_func=None):
        super(Sequential, self).__init__(trainable=False)
        self._modules = OrderedDict()

        if modules is not None:
            if isinstance(modules, OrderedDict):
                for name, module in modules.items():
                    self.add(module, name)

            elif isinstance(modules, list):
                for idx, module in enumerate(modules):
                    name = getattr(module, 'name')
                    mod_name = name if name is not None else str(idx)
                    self.add(module, mod_name)
            else:
                raise TypeError('Given parameter {} is not valid '.format(type(modules)))

        self._loss = loss_func

    def add(self, module: Module, name=None):
        if module is None or isinstance(module, Loss):
            raise ValueError('Given object type is Loss is not valid !')
        if module is None or not isinstance(module, Module):
            raise ValueError('Given object type {} is not Module '.format(type(module)))

        if name is None or len(name) == 0:
            name = str(len(self._modules)) + "_" + module.__class__.__name__

        print("Added Module Name: {} ".format(name))
        self._modules[name] = module
        act_layer = getattr(module, 'activation', None)

        if act_layer is not None:
            self.add(act_layer, getattr(act_layer, 'name', None))

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor=False):

        is_test = target is None
        out = input

        for module in self._modules.values():
            if is_test and isinstance(module, Dropout):
                continue
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

    def accuracy(self, y_pred: torch.FloatTensor, y_target: torch.FloatTensor):
        return (y_pred == y_target).type(torch.FloatTensor).mean()

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