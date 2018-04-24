import torch
from exceptions import ShapeException
from .module import Module


# TODO: tests
class ActivationModule(Module):

    def __init__(self):
        super(ActivationModule, self).__init__(trainable=False)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def dim_check(self, tensor_context: str, tensor_: torch.FloatTensor, dim: int):
        if tensor_.dim() != dim:
            raise ShapeException('Given {} dimension({}), required dimension is {}'
                                 .format(tensor_context, tensor_.dim(), dim))


class ReLU(ActivationModule):

    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None

    def forward(self, tensor_in: torch.FloatTensor):

        super().dim_check("forward input", tensor_in, 2)

        self.input = tensor_in
        tensor_out = torch.max(tensor_in, torch.zeros(tensor_in.size()))

        return tensor_out

    def backward(self, gradwrtoutput: torch.FloatTensor):

        if gradwrtoutput.dim() != 2:
            raise ShapeException('Given gradwrtoutput dimension({}), required dimension is 2'
                                 .format(gradwrtoutput.dim()))
        if self.input is None:
            raise ValueError('Input is not set, backward cannot be called !')

        gradwrtoutput[self.input <= 0] = 0

        return gradwrtoutput


class Sigmoid(ActivationModule):

    # training is problematic, idk why
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output = None

    def forward(self, tensor_in: torch.FloatTensor):
        super().dim_check("forward input", tensor_in, 2)
        tensor_out = torch.sigmoid(tensor_in)
        # tensor_out = 1/(1+torch.exp(-tensor_in))

        self.output = tensor_out
        return tensor_out

    def backward(self, gradwrtoutput: torch.FloatTensor):

        if gradwrtoutput.dim() != 2:
            raise ShapeException('Given gradwrtoutput dimension({}), required dimension is 2'
                                 .format(gradwrtoutput.dim()))

        dsigmoid = self.output * (1 - self.output)

        # return torch.mm(gradwrtoutput, dsigmoid.transpose(0,1))
        return gradwrtoutput * dsigmoid


# TODO: implement Softmax
class Softmax(ActivationModule):

    def __init__(self):
        super(Softmax, self).__init__()
        self.output = None

    def forward(self, tensor_in: torch.FloatTensor):
        super().dim_check("forward input", tensor_in, 2)

        row_maxs, _ = tensor_in.max(1)
        x = torch.exp(tensor_in - row_maxs.repeat(tensor_in.shape[1], 1).transpose(0, 1))
        tensor_out = x / x.sum(1).repeat(tensor_in.shape[1], 1).transpose(0, 1)

        self.output = tensor_out

        return tensor_out

    def backward(self, gradwrtoutput: torch.FloatTensor):
        # not implemented
        if gradwrtoutput.dim() == 1:
            gradwrtoutput = gradwrtoutput.unsqueeze(0)

        dsoftmax = self.output * (1 - self.output)

        return gradwrtoutput * dsoftmax


class Tanh(ActivationModule):

    def __init__(self):
        super(Tanh, self).__init__()
        self.output = None

    def forward(self, tensor_in: torch.FloatTensor):
        super().dim_check("forward input", tensor_in, 2)

        tensor_out = torch.tanh(tensor_in)
        self.output = tensor_out
        return tensor_out

    def backward(self, gradwrtoutput: torch.FloatTensor):

        super().dim_check("backward grad", gradwrtoutput, 2)
        dtanh = (1 + self.output) * (1 - self.output)

        # return torch.mm(gradwrtoutput, dtanh.transpose(0,1))
        return gradwrtoutput * dtanh
