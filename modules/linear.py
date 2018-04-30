import torch
from collections import OrderedDict
from exceptions import ShapeException
import numpy as np
from .module import Module


class Linear(Module):

    def __init__(self, in_num, out_num, is_bias=True, is_trainable=True, activation: Module=None, name=None):
        super(Linear, self).__init__(trainable=is_trainable)
        self.in_num = in_num
        self.out_num = out_num

        self.input = None
        self._params, self._grads = OrderedDict(), OrderedDict()

        self._params["weight"] = torch.FloatTensor(in_num, out_num)
        self._grads["weight"] = None
        self._params["bias"], self._grads["bias"] = None, None

        self.is_bias = is_bias
        if is_bias:
            self._params["bias"] = torch.FloatTensor(out_num)
            self._grads["bias"] = None

        self._initialize_parameters()
        self._activation = activation
        self._name = name

    def _initialize_parameters(self, is_xavier_initialization=True):

        denominator = (self.in_num + self.out_num) if is_xavier_initialization else self.in_num
        std = np.sqrt(2.0 / denominator)

        self._params["weight"].uniform_(-std, std)
        if self.is_bias:
            self._params["bias"].uniform_(-std, std)

    def forward(self, tensor_in: torch.FloatTensor):

        self.dim_check("input tensor forward@Linear", tensor_in, dim=2)

        self.input = tensor_in
        tensor_out = torch.mm(tensor_in, self._params["weight"])

        if self.is_bias:
            tensor_out += self._params["bias"]

        return tensor_out

    def backward(self, gradwrtoutput: torch.FloatTensor):

        self.dim_check("gradwrtoutput backward@Linear", gradwrtoutput, dim=2)

        if self.is_bias:
            self._grads["bias"] = torch.mv(gradwrtoutput.transpose(0, 1), torch.ones(gradwrtoutput.shape[0]))

        self._grads["weight"] = torch.mm(self.input.transpose(0, 1), gradwrtoutput)
        return torch.mm(gradwrtoutput, self._params["weight"].transpose(0, 1))

    @property
    def name(self):
        return self._name

    @property
    def activation(self):
        return self._activation

    @property
    def params(self):
        keys = ['weight', 'bias']
        for key in keys:
            if self._params[key] is not None:
                yield key, self._params[key]

    def grads(self, key_name):
        return self._grads[key_name]

    def set_param(self, name, value):

        if name not in self._params:
            raise ValueError('Given key: {} is not valid !'.format(name))

        if not isinstance(value, torch.FloatTensor):
            raise TypeError('Required type is torch.FloatTensor, given {}'.format(type(value)))

        if value.shape != self._params[name].shape:
            raise ShapeException('Given shape ({}) does not match with required ({})'.
                                 format(value.shape, self._params[name]))

        self._params[name] = value

from torch.optim.sgd import SGD