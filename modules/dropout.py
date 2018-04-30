import numpy as np
import torch
from .module import Module


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
