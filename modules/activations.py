import torch
from .module import Module
from .module import require_dimension, require_not_none


class ActivationModule(Module):
    """
    All of the activation layer must extend this class
    implementing forward and backward layer
    """
    def __init__(self):
        """
        Activation layers do not have any parameter to learn
        Thus, trainable attribute is set to False at start
        """
        super(ActivationModule, self).__init__(trainable=False)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def __str__(self):
        return self.name


class ReLU(ActivationModule):
    """
    ReLU activation function implementation: max(0, x)
    """
    def __init__(self):
        super(ReLU, self).__init__()
        # value to be stored in forward pass for later use in backward pass
        self.input = None

    @require_dimension(dim=2)
    def forward(self, tensor_in: torch.FloatTensor):
        # store input for later use in backward pass
        self.input = tensor_in
        # vectorize version of max(0, x), which is elementwise
        tensor_out = torch.max(tensor_in, torch.zeros(tensor_in.size()))

        return tensor_out

    @require_not_none('input')
    @require_dimension(dim=2)
    def backward(self, gradwrtoutput: torch.FloatTensor):
        if self.input is None:
            raise ValueError('Input is not set, backward cannot be called !')
        
        # backward pass of relu, use stored input of forward pass as mask
        gradwrtoutput[self.input <= 0] = 0
        # set stored value to None
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
        # find max element of each row
        row_maxs, _ = tensor_in.max(1)
        # subtract max element of row from each element of corresponding row to prevent overflow
        # then take elementwise exp
        x = torch.exp(tensor_in - row_maxs.repeat(tensor_in.shape[1], 1).transpose(0, 1))
        # normalize with sum of each element in row to get probability distribution
        tensor_out = x / x.sum(1).repeat(tensor_in.shape[1], 1).transpose(0, 1)
        
        return tensor_out

    @require_dimension(dim=2)
    def backward(self, gradwrtoutput: torch.FloatTensor):
        # softmax class is assumed to be used in the last layer with cross-entropy-loss object
        # gradient of softmax and cross-entropy together is handled in backward pass of cross-entropy loss
        # thus, backward pass of softmax is identity
        return gradwrtoutput


class Tanh(ActivationModule):
    """
    tanh activation function: (e^z - e^(-z)) / (e^z + e^(-z))
    """
    def __init__(self):
        super(Tanh, self).__init__()
        # value to be stored in forward pass for later use in backward pass
        self.output = None

    @require_dimension(dim=2)
    def forward(self, tensor_in: torch.FloatTensor):
        # use existing tanh tensor function of pytorch
        tensor_out = torch.tanh(tensor_in)
        # store output for later use in backward pass
        self.output = tensor_out
        return tensor_out

    @require_not_none('output')
    @require_dimension(dim=2)
    def backward(self, gradwrtoutput: torch.FloatTensor):
        # gradient of tanh wrt to input, using stored input of forward pass
        dtanh = (1 + self.output) * (1 - self.output)
        # set stored value to None
        self.output = None
        
        # returned value is elementwise product of incoming gradient and gradient of tanh wrt to its input
        return gradwrtoutput * dtanh

