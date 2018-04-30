from .activations import *
from collections import OrderedDict
from exceptions import ShapeException
import torch

_nonlinear_funcs = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'tanh': Tanh
}


class Linear(Module):

    def __init__(self, in_num, out_num,
                 is_bias=True,
                 activation=None,
                 name=None):

        super(Linear, self).__init__(trainable=True)
        self.in_num = in_num
        self.out_num = out_num

        self.input = None
        self._params, self._grads = OrderedDict(), OrderedDict()

        self._params["weight"] = torch.FloatTensor(in_num, out_num)
        self._grads["weight"] = None
        self._params["bias"], self._grads["bias"] = None, None

        self._name = name
        self.is_bias = is_bias
        if is_bias:
            self._params["bias"] = torch.FloatTensor(out_num)
            self._grads["bias"] = None

        # self.do_dropout = self.p_dropout > 0
        self._initialize_parameters()
        if isinstance(activation, str):
            try:
                self._activation = _nonlinear_funcs[activation]()
            except KeyError:
                print("Given activation function {} is invalid".format(activation))
                print('Available activation functions are following:\n'.format(str(_nonlinear_funcs.keys())))
        elif isinstance(activation, Module):
            self._activation = activation

    def _initialize_parameters(self, is_xavier_initialization=True):

        denominator = (self.in_num + self.out_num) if is_xavier_initialization else self.in_num
        std = np.sqrt(2.0 / denominator)

        self._params["weight"].uniform_(-std, std)
        if self.is_bias:
            self._params["bias"].uniform_(-std, std)

    def forward(self, tensor_in: torch.FloatTensor, do_dropout=True):

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

    @property
    def grads(self):
        return self._grads

    def set_param(self, name, value):

        if name not in self._params:
            raise ValueError('Given key: {} is not valid !'.format(name))

        if not isinstance(value, torch.FloatTensor):
            raise TypeError('Required type is torch.FloatTensor, given {}'.format(type(value)))

        if value.shape != self._params[name].shape:
            raise ShapeException('Given shape ({}) does not match with required ({})'.
                                 format(value.shape, self._params[name]))

        self._params[name] = value


class Dropout(Module):

    def __init__(self, prob=0., name=None):
        self.prob = 1-prob
        self.mask = None
        super(Dropout, self).__init__(trainable=False, name=name)

    def forward(self, tensor_in: torch.FloatTensor):
        self.dim_check("input tensor forward@Linear", tensor_in, dim=2)
        tmp = np.random.binomial(1, self.prob, size=tensor_in.shape) / self.prob

        self.mask = torch.from_numpy(tmp).type(torch.FloatTensor)

        return tensor_in * self.mask

    def backward(self, gradwrtoutput):
        grad = gradwrtoutput * self.mask
        self.mask = None
        return grad
