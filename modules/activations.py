import math
import torch
from .module import Module
from .module import require_dimension, require_not_none


class ActivationModule(Module):

    def __init__(self):
        super(ActivationModule, self).__init__(trainable=False)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def __str__(self):
        return self.name


class ReLU(ActivationModule):

    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None

    @require_dimension(dim=2)
    def forward(self, tensor_in: torch.FloatTensor):
        self.input = tensor_in
        tensor_out = torch.max(tensor_in, torch.zeros(tensor_in.size()))

        return tensor_out

    @require_not_none('input')
    @require_dimension(dim=2)
    def backward(self, gradwrtoutput: torch.FloatTensor):
        if self.input is None:
            raise ValueError('Input is not set, backward cannot be called !')

        gradwrtoutput[self.input <= 0] = 0
        self.input = None
        return gradwrtoutput


class Softmax(ActivationModule):
    """
    This class can only be used in the last layer with cross-entropy-loss object.
    Cross-entropy-backward covers both its and this backward.
    Therefore, Softmax backward just passes what it passed to it.
    """
    def __init__(self):
        super(Softmax, self).__init__()

    @require_dimension(dim=2)
    def forward(self, tensor_in: torch.FloatTensor):
        row_maxs, _ = tensor_in.max(1)
        x = torch.exp(tensor_in - row_maxs.repeat(tensor_in.shape[1], 1).transpose(0, 1))
        tensor_out = x / x.sum(1).repeat(tensor_in.shape[1], 1).transpose(0, 1)
        return tensor_out

    @require_dimension(dim=2)
    def backward(self, gradwrtoutput: torch.FloatTensor):
        return gradwrtoutput


class Tanh(ActivationModule):
    """

    """
    def __init__(self):
        super(Tanh, self).__init__()
        self.output = None

    @require_dimension(dim=2)
    def forward(self, tensor_in: torch.FloatTensor):
        tensor_out = torch.tanh(tensor_in)
        self.output = tensor_out
        return tensor_out

    @require_not_none('output')
    @require_dimension(dim=2)
    def backward(self, gradwrtoutput: torch.FloatTensor):
        dtanh = (1 + self.output) * (1 - self.output)
        self.output = None
        return gradwrtoutput * dtanh

