from .layers import Dropout
from .losses import Loss
from collections import OrderedDict
from .activations import *
import torch


class Sequential(Module):

    def __init__(self, modules=None, loss_func=None):
        super(Sequential, self).__init__(trainable=False)
        self._modules = OrderedDict()
        self.layer_sizes = []
        if modules is not None:
            if isinstance(modules, list):
                for idx, module in enumerate(modules):
                    self.add(module)
            else:
                raise TypeError('Given parameter {} is not valid '.format(type(modules)))

        self._loss = loss_func

    def add(self, module: Module):

        if module is None or isinstance(module, Loss):
            raise ValueError('Given object type is Loss is not valid !')
        if module is None or not isinstance(module, Module):
            raise ValueError('Given object type {} is not Module '.format(type(module)))

        if hasattr(module, 'input_size'):
            if module.input_size is None and len(self.layer_sizes) == 0:
                raise ValueError('First module must specify input size in Linear layer !')
            if module.input_size is not None and \
                    (len(self.layer_sizes) != 0 and module.input_size == self.layer_sizes[-1]):
                raise ValueError('Given input size {} is not compatible with previous layer size: {}'
                                 .format(module.input_size,  self.layer_sizes[-1]))

            if module.input_size is None:
                module.input_size = self.layer_sizes[-1]

            module.initialize()
            self.layer_sizes.append(getattr(module, 'output_size'))

        name = str(len(self._modules)) + "_" + module.__class__.__name__
        self._modules[name] = module

        print("Added Module Name: {} ".format(name))
        try:
            self.add(getattr(module, 'activation'))
        except AttributeError:
            pass

    def forward(self, input: torch.FloatTensor, target: torch.FloatTensor=False, train=False):

        out = input

        for module in self._modules.values():
            if isinstance(module, Dropout) and not train:
                continue
            out = module.forward(out)

        return out if target is None else (out, self._loss.forward(out, target, val))

    def backward(self):
        gradwrtoutputt = self._loss.backward()
        for module in reversed(list(self._modules.values())):
            gradwrtoutputt = module.backward(gradwrtoutputt)

    def predict(self, x_test: torch.FloatTensor):
        if not torch.is_tensor(x_test):
            raise ValueError('Given x_test parameter must be torch.Tensor !')

        y_pred = self.forward(x_test)
        print(y_pred)
        return y_pred.max(1)[1]

    def evaluate(self, x_test: torch.FloatTensor, y_test: torch.FloatTensor):
        if not torch.is_tensor(y_test):
            raise ValueError('Given x_test parameter must be torch.Tensor !')
        if len(y_test.shape) != 1:
            raise ShapeException('Given y_test with shape {} is not valid, required [nRows] '.format(y_test.shape))

        y_pred = self.predict(x_test)

        return Sequential.accuracy(y_pred, y_test)

    @staticmethod
    def accuracy(y_pred, y_test):
        return (y_pred == y_test).type(torch.FloatTensor).mean()

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
