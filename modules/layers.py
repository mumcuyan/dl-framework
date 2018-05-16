from collections import OrderedDict
import torch
import numpy as np

from .activations import ReLU, Softmax, Tanh
from .module import Module
from .module import require_dimension, require_not_none
from exceptions import ShapeException


_nonlinear_funcs = {
    'relu': ReLU,
    'softmax': Softmax,
    'tanh': Tanh
}


def require_initialization(f):
    """
    This function is used for weights, whether it is initialized or not
    if it is not initialized, training cannot be started
    :param f: function which will use self._params['weight']
    :return: if it is not initialized, it will initialize
    """

    def wrapper(self, *args):
        if self._params['weight'] is None:
            self.initialize()

        return f(self, *args)
    return wrapper


class Linear(Module):

    def __init__(self, out, input_size=None,
                 is_bias=True,
                 activation=None, name=None):
        """
        :param out: unit size of layer
        :param input_size: unit size of previous layer
        :param is_bias: boolean flag to have bias vector or not
        :param activation: is a string, each mapping to an object defined in modules.activations
        :param name: user-specified name can be passed to each Linear, however, this must be unique over entire network
        """
        super(Linear, self).__init__(trainable=True, name=name)
        self.in_num = input_size
        self.out_num = out
        self.input = None
        self.is_bias = is_bias

        self._params, self._grads = OrderedDict(), OrderedDict()
        self._grads["bias"], self._grads['weight'] = None, None
        self._params['bias'], self._params['weight'] = None, None

        if isinstance(activation, str):
            try:
                self._activation = _nonlinear_funcs[activation]()  # passed activation function name should be valid
            except KeyError:
                print("Given activation function {} is invalid".format(activation))
                print('Available activation functions are following:\n'.format(str(_nonlinear_funcs.keys())))

        elif isinstance(activation, Module):  # activation can be also passed as an object
            self._activation = activation

    def initialize(self, is_xavier_initialization=True):
        """
        :param is_xavier_initialization:  boolean flag of whether applying xavier initialization or not
        :return: initialized version of weight and bias
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
        given input tensor multiplies with weight matrix and returns to next laye
        :param tensor_in: calculated matrix passed by previous layer, must be 2D
        :return: computed tensor by multiplication (addiiton for bias)
        """

        self.input = tensor_in
        tensor_out = torch.mm(tensor_in, self._params["weight"])

        if self.is_bias:
            tensor_out += self._params["bias"]

        return tensor_out

    @require_not_none('input')
    @require_dimension(dim=2)
    def backward(self, gradwrtoutput: torch.FloatTensor):
        """
        given gradient wrt output, it calculates gradient of both bias and weights and returns to previous layer
        :param gradwrtoutput:
        :return:
        """
        if self.is_bias:
            self._grads["bias"] = torch.mv(gradwrtoutput.transpose(0, 1), torch.ones(gradwrtoutput.shape[0]))

        self._grads["weight"] = torch.mm(self.input.transpose(0, 1), gradwrtoutput)
        self.input = None # reset

        return torch.mm(gradwrtoutput, self._params["weight"].transpose(0, 1))

    def set_param(self, name, value):
        """
        setting parameter for gradient update, this function will be used by optimizer
        to update Linear's weight

        :param name: either 'bias' or 'weight'
        :param value: torch.FloatTensor
        """

        if name not in self._params:
            raise ValueError('Given key: {} is not valid !'.format(name))

        if not isinstance(value, torch.FloatTensor):
            raise TypeError('Required type is torch.FloatTensor, given {}'.format(type(value)))

        if self._params[name] is not None and value.shape != self._params[name].shape:
            raise ShapeException('Given shape ({}) does not match with required ({})'.
                                 format(value.shape, self._params[name]))
        
        self._params[name] = value

    def __str__(self):
        """
        each Module implements __str__ function
        This is used for printing out overview of the module for user
        :return: string version of description Linear object (including name, input_size, output_size)
        """
        return self.name + "\t\t (" + str(self.in_num) + ")" + "\t\t\t (" + str(self.out_num) + ")"

    @property
    def activation(self):
        # if activation is not set in initialization, never be set in the future
        # it is designed for user-friendly interface for model generation
        if self._activation is None:
            raise AttributeError()
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

    """
    following names are alias (properties) for Linear class attributes
    """
    name = property(get_name, set_name)
    input_size = property(fget=get_input_size, fset=set_input_size)
    output_size = property(fget=get_output_size, fset=set_output_size)


class Dropout(Module):
    """
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    """
    def __init__(self, prob=0., name=None):
        self.prob = 1-prob  # keep reverse prob (indicating prob of keeping the neuron)
        self.mask = None
        super(Dropout, self).__init__(trainable=False, name=name)

    @require_dimension(dim=2)
    def forward(self, tensor_in: torch.FloatTensor):
        prob_tensor = torch.FloatTensor(tensor_in.shape).fill_(self.prob)
        self.mask = torch.bernoulli(prob_tensor) / self.prob  # inverted Dropout implemented

        return tensor_in * self.mask

    @require_not_none('mask')  # mask must not be None, meaning forward must be called before each call of backward
    def backward(self, gradwrtoutput):
        grad = gradwrtoutput * self.mask
        self.mask = None  # reset it to make sure that the last mask is
        return grad

    def __str__(self):
        return "{} -p: {: .2f}".format(self.name, 1-self.prob)
