from collections import OrderedDict
from .module import Module
import torch


class Sequential(Module):

    def __init__(self, modules=None):
        super(Sequential, self).__init__()
        self._modules = OrderedDict()

        if isinstance(modules, OrderedDict):
            for name, module in modules.items():
                self.add_module(module, name)
        elif isinstance(modules, list):
            for idx, module in enumerate(modules):
                self.add_module(module, str(idx))
        else:
            raise TypeError('Given parameter {} is not valid '.format(type(modules)))

        self.output = None

    def add_module(self, module: Module, name):

        if module is None or not isinstance(module, Module):
            raise ValueError('Given object type {} is not Module '.format(type(module)))
        if name is None or len(name) == 0:
            raise ValueError('Given name {} is not valid'.format(name))

        self._modules[name] = module

    def forward(self, input: torch.FloatTensor):

        tmp_input = input
        if tmp_input.dim() == 1:
            tmp_input = tmp_input.unsqueeze(0)

        for module in self._modules.values():
            tmp_input = module.forward(tmp_input)

        self.output = tmp_input

        # TODO ?, predict
        _, self.prediction = self.output.max(1)

        return self.output

    def backward(self):

        gradwrtoutputt = torch.FloatTensor([[1]])

        for module in reversed(list(self._modules.values())):
            gradwrtoutputt = module.backward(gradwrtoutputt)

        self.output = None
        self.prediction = None

        return gradwrtoutputt

    def predict(self, input):

        if self.prediction is None:
            if self.output is None:
                pass
                # self.forward_without_loss(input)
            else:
                _, self.prediction = self.output.max(1)

        return self.prediction

    def accuracy(self, input, target):
        self.predict(input)
        return (self.prediction == target).type(torch.FloatTensor).mean(), self.prediction

    def print_model(self):
        for module in self._modules:
            print(module)

    @property
    def trainable_modules(self):
        for module in self._modules.values():
            if module.trainable:
                yield module