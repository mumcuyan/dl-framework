from collections import OrderedDict
import torch
import numpy as np

from torch.distributions import __all__
from .activations import ReLU, Softmax, Tanh
from .module import Module
from .module import require_dimension
from exceptions import ShapeException


_nonlinear_funcs = {
    'relu': ReLU,
    'softmax': Softmax,
    'tanh': Tanh
}


def require_initialization(f):
    def wrapper(self, *args):
        if self._params['weight'] is None:
            self.initialize()

        return f(self, *args)
    return wrapper


class Linear(Module):

    def __init__(self, out, input_size=None,
                 is_bias=True,
                 activation=None):

        super(Linear, self).__init__(trainable=True)
        self.in_num = input_size
        self.out_num = out
        self.input = None
        self.is_bias = is_bias

        self._params, self._grads = OrderedDict(), OrderedDict()
        self._grads["bias"], self._grads['weight'] = None, None
        self._params['bias'], self._params['weight'] = None, None

        if isinstance(activation, str):
            try:
                self._activation = _nonlinear_funcs[activation]()
            except KeyError:
                print("Given activation function {} is invalid".format(activation))
                print('Available activation functions are following:\n'.format(str(_nonlinear_funcs.keys())))
        elif isinstance(activation, Module):
            self._activation = activation

    def initialize(self, is_xavier_initialization=True):
        """
        :param is_xavier_initialization:
        :return:
        """
        if self.in_num is None:
            raise ValueError('Input layer size cannot be None !')

        self._params["weight"] = torch.FloatTensor(self.in_num, self.out_num)
        self._params["bias"] = torch.FloatTensor(self.out_num) if self.is_bias else None

        denominator = (self.in_num + self.out_num) if is_xavier_initialization else self.in_num
        std = np.sqrt(2.0 / denominator)

        self._params["weight"].uniform_(-std, std)
        if self.is_bias:
            self._params["bias"].uniform_(-std, std)

    @require_initialization
    @require_dimension(dim=2)
    def forward(self, tensor_in: torch.FloatTensor):
        """
        :param tensor_in:
        :return:
        """

        self.input = tensor_in
        tensor_out = torch.mm(tensor_in, self._params["weight"])

        if self.is_bias:
            tensor_out += self._params["bias"]

        return tensor_out

    @require_dimension(dim=2)
    def backward(self, gradwrtoutput: torch.FloatTensor):
        """
        :param gradwrtoutput:
        :return:
        """
        if self.is_bias:
            self._grads["bias"] = torch.mv(gradwrtoutput.transpose(0, 1), torch.ones(gradwrtoutput.shape[0]))

        self._grads["weight"] = torch.mm(self.input.transpose(0, 1), gradwrtoutput)
        return torch.mm(gradwrtoutput, self._params["weight"].transpose(0, 1))

    def set_param(self, name, value):

        if name not in self._params:
            raise ValueError('Given key: {} is not valid !'.format(name))

        if not isinstance(value, torch.FloatTensor):
            raise TypeError('Required type is torch.FloatTensor, given {}'.format(type(value)))

        if value.shape != self._params[name].shape:
            raise ShapeException('Given shape ({}) does not match with required ({})'.
                                 format(value.shape, self._params[name]))

        self._params[name] = value

    @property
    def activation(self):
        if self._activation is None:
            raise AttributeError()  # if activation is not set in initialization, never be set in the future
        return self._activation

    @property
    def params(self):
        keys = ['weight', 'bias']
        for key in keys:
            if self._params[key] is not None:
                yield key, self._params[key]

    @property
    def grads(self):
        return self._grads

    def set_name(self, value):
        self._name = value

    def get_name(self):
        return self._name

    def set_input_size(self, value):
        self.in_num = value

    def get_input_size(self):
        return self.in_num

    def get_output_size(self):
        return self.out_num

    def set_output_size(self, value):
        self.out_num = value

    name = property(get_name, set_name)
    input_size = property(fget=get_input_size, fset=set_input_size)
    output_size = property(fget=get_output_size, fset=set_output_size)


class Dropout(Module):

    def __init__(self, prob=0., name=None):
        self.prob = 1-prob
        self.mask = None
        super(Dropout, self).__init__(trainable=False, name=name)

    @require_dimension(dim=2)
    def forward(self, tensor_in: torch.FloatTensor):
        prob_tensor = torch.FloatTensor(tensor_in.shape).fill_(self.prob)
        self.mask = torch.bernoulli(prob_tensor) / self.prob

        return tensor_in * self.mask

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput * self.mask
        self.mask = None
        return grad
