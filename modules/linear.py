import torch
from collections import OrderedDict
import numpy as np
from .module import Module


class Linear(Module):

    def __init__(self, in_num, out_num, is_bias=True, is_trainable=True):
        super(Linear, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.is_trainable = is_trainable

        self.input = None

        self.params = OrderedDict()
        self.grads = OrderedDict()

        self.params["weight"] = torch.FloatTensor(in_num, out_num)
        self.grads["weight"] = None

        self.is_bias = is_bias

        if is_bias:
            self.params["bias"] = torch.FloatTensor(out_num)
            self.grads["bias"] = None
        else:
            self.params["bias"] = None
            self.grads["bias"] = None

        self.initialize_parameters()

    def initialize_parameters(self, is_xavier_initialization=True):

        if is_xavier_initialization:
            std = np.sqrt(2.0 / (self.in_num + self.out_num))
        else:
            std = np.sqrt(1.0 / (self.in_num))

        self.params["weight"].uniform_(-std, std)
        if self.is_bias:
            self.params["bias"].uniform_(-std, std)

    def set_parameters(self, new_weight, new_bias=None):

        if self.params["weight"].shape == new_weight.shape:
            self.params["weight"] = new_weight
            self.grads["weight"] = None
        if new_bias is not None and self.params["weight"].shape[-1] == new_bias.shape[0]:
            self.params["bias"] = new_bias
            self.grads["bias"] = None

    def forward(self, tensor_in: torch.FloatTensor):

        if tensor_in.dim() == 1:
            print("@tensor_in.dim(): " + tensor_in.dim())
            tensor_in = tensor_in.unsqueeze(0)

        self.input = tensor_in
        tensor_out = torch.mm(tensor_in, self.params["weight"])

        if self.is_bias:
            tensor_out += self.params["bias"]

        return tensor_out

    def backward(self, gradwrtoutput: torch.FloatTensor):

        if gradwrtoutput.dim() == 1:
            print("@gradwrtoutput.dim() ", gradwrtoutput.dim())
            gradwrtoutput = gradwrtoutput.unsqueeze(0)

        if self.is_bias:
            self.grads["bias"] = torch.mv(gradwrtoutput.transpose(0, 1), torch.ones(gradwrtoutput.shape[0]))

        self.grads["weight"] = torch.mm(self.input.transpose(0, 1), gradwrtoutput)

        return torch.mm(gradwrtoutput, self.params["weight"].transpose(0, 1))

    def param(self):
        return [(self.params["weight"], self.grads["weight"]), (self.params["bias"], self.grads["bias"])]

    def set_param(self, name, value):

        if name in self.params and self.params[name].shape == value.shape:
            self.params[name] = value
