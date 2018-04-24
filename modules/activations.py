import torch


# TODO: put is_trainable as False in ActivationModule
# TODO: activationlar'a test yazilacak


class ActivationModule:

    def __init__(self):
        pass

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    """
    def param(self):
        return []
    """
    def set_param(self, name, value):
        pass


class ReLU(ActivationModule):

    def __init__(self, is_trainable=False):
        super(ReLU, self).__init__()

        self.input = None
        self.is_trainable = is_trainable

    def forward(self, tensor_in):

        if tensor_in.dim() == 1:
            tensor_in = tensor_in.unsqueeze(0)

        self.input = tensor_in

        tensor_out = torch.max(tensor_in, torch.zeros(tensor_in.size()))

        return tensor_out

    def backward(self, gradwrtoutput):

        if gradwrtoutput.dim() == 1:
            gradwrtoutput = gradwrtoutput.unsqueeze(0)

        gradwrtoutput[self.input <= 0] = 0

        return gradwrtoutput


class Sigmoid(ActivationModule):

    # training is problematic, idk why
    def __init__(self, is_trainable=False):
        super(Sigmoid, self).__init__()

        self.output = None
        self.is_trainable = is_trainable

    def forward(self, tensor_in):

        if tensor_in.dim() == 1:
            tensor_in = tensor_in.unsqueeze(0)

        tensor_out = torch.sigmoid(tensor_in)
        # tensor_out = 1/(1+torch.exp(-tensor_in))

        self.output = tensor_out
        return tensor_out

    def backward(self, gradwrtoutput):

        if gradwrtoutput.dim() == 1:
            gradwrtoutput = gradwrtoutput.unsqueeze(0)

        dsigmoid = self.output * (1 - self.output)

        # return torch.mm(gradwrtoutput, dsigmoid.transpose(0,1))
        return gradwrtoutput * dsigmoid


class Softmax(ActivationModule):

    def __init__(self, is_trainable=False):
        super(Softmax, self).__init__()

        self.output = None
        self.is_trainable = is_trainable

    def forward(self, tensor_in: torch.FloatTensor):

        if tensor_in.dim() == 1:
            tensor_in = tensor_in.unsqueeze(0)

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

        # return torch.mm(gradwrtoutput, dtanh.transpose(0,1))
        return gradwrtoutput * dsoftmax


class Tanh(ActivationModule):

    def __init__(self, is_trainable=False):
        super(Tanh, self).__init__()

        self.output = None
        self.is_trainable = is_trainable

    def forward(self, tensor_in):

        if tensor_in.dim() == 1:
            tensor_in = tensor_in.unsqueeze(0)

        tensor_out = torch.tanh(tensor_in)

        self.output = tensor_out

        return tensor_out

    def backward(self, gradwrtoutput):

        if gradwrtoutput.dim() == 1:
            gradwrtoutput = gradwrtoutput.unsqueeze(0)

        dtanh = (1 + self.output) * (1 - self.output)

        # return torch.mm(gradwrtoutput, dtanh.transpose(0,1))
        return gradwrtoutput * dtanh
