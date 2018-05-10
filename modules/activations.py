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

        if math.isnan(tensor_out.sum()):
            print("@relu tensor_in: {}".format(tensor_in))
            print("@relu tensor_out: {}".format(tensor_out))
            exit(1)

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


"""
    Softmax backward layer
    # https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    # numpy conversion, might not work
    def backward(self, gradwrtoutput: torch.FloatTensor):
    
    super().dim_check("backward grad @ Softmax", gradwrtoutput, 2)
    softmax_np = self.output.numpy()
    softmax_np = softmax_np.reshape(-1, 1)
    dsoftmax = torch.from_numpy(np.diagflat(softmax_np) - np.dot(softmax_np, softmax_np.T))

    return gradwrtoutput * dsoftmax
"""

"""
class Sigmoid(ActivationModule):

    # TODO: training is problematic, idk why
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output = None

    def forward(self, tensor_in: torch.FloatTensor):
        super().dim_check("forward input @ Sigmoid", tensor_in, 2)
        tensor_out = torch.sigmoid(tensor_in)
        # tensor_out = 1/(1+torch.exp(-tensor_in))
        self.output = tensor_out
        return tensor_out

    def backward(self, gradwrtoutput: torch.FloatTensor):
        super().dim_check("backward grad @ Sigmoid", gradwrtoutput, 2)

        dsigmoid = self.output * (1 - self.output)
        self.output = None  # reset output None
        # return torch.mm(gradwrtoutput, dsigmoid.transpose(0,1))
        return gradwrtoutput * dsigmoid

"""