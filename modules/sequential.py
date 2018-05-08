from .layers import Dropout
from utils import one_hot2label
from .losses import Loss
from collections import OrderedDict
from .activations import *
import torch
import collections

from exceptions import InputSizeNotFoundError, NotCompatibleError


class Sequential(Module):

    def __init__(self, modules=None, loss_func=None):
        super(Sequential, self).__init__(trainable=False)
        self._modules = OrderedDict()
        self.layer_sizes = []
        if modules is not None:
            if isinstance(modules, collections.Sequence):
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
                raise InputSizeNotFoundError('First module must specify input size in Linear layer !')
            if module.input_size is not None and \
                    (len(self.layer_sizes) != 0 and module.input_size != self.layer_sizes[-1]):
                raise NotCompatibleError('Given input size {} is not compatible with previous layer size: {}'
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

    def forward(self, x_input: torch.FloatTensor, y_input: torch.FloatTensor=None):

        train = y_input is not None
        y_pred = x_input
        for module in self._modules.values():
            if isinstance(module, Dropout) and not train:
                continue
            y_pred = module.forward(y_pred)

        if train:
            self._loss.forward(y_pred, y_input)

        return y_pred

    def backward(self):
        gradwrtoutputt = self._loss.backward()
        for module in reversed(list(self._modules.values())):
            gradwrtoutputt = module.backward(gradwrtoutputt)

    def predict(self, x_test: torch.FloatTensor, convert_label=False):
        """
        :param x_test:
        :param convert_label: boolean flag whether to convert prob dist to label
        :return: return N x 2 size tensor as a y_prediction
        """
        if not torch.is_tensor(x_test):
            raise TypeError('Given x_test parameter must be torch.Tensor !')

        y_pred = self.forward(x_test)
        return y_pred if not convert_label else y_pred.max(1)[1]

    def evaluate(self, x_test: torch.FloatTensor, y_test: torch.FloatTensor, return_pred=False):
        """
        To get accuracy, loss and prediction of x_test based on trained model, use this function.
        If you do not have correct answers please refer to .predict function defined for this object.

        :param x_test: [N x feature_dim] torch.FloatTensor
        :param y_test: [N x class_num] torch.FloatTensor (one-hot encoded)
        :param return_pred: flag var for returning labels for each test data point --> [N x 1] torch.FloatTensor
        :return: accuracy of prediction and loss value, based on return_pred, predictions as labels
        """
        if not torch.is_tensor(y_test):
            raise TypeError('Given x_test parameter must be torch.Tensor !')

        y_pred = self.predict(x_test)
        loss_val = self._loss(y_pred, y_test)

        acc_val = self.accuracy(one_hot2label(y_pred), one_hot2label(y_test))

        if not return_pred:
            return acc_val, loss_val
        else:
            return acc_val, loss_val, y_pred.max(1)[1]

    @staticmethod
    def accuracy(y_pred, y_test):
        if not torch.is_tensor(y_test):
            raise TypeError('Given x_test parameter must be torch.Tensor !')

        return (y_pred == y_test).type(torch.FloatTensor).mean()

    def print_model(self):
        for name, module in self._modules.items():
            print("Name: {}".format(name))

    @property
    def module_names(self):
        return list(self._modules.keys())

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
