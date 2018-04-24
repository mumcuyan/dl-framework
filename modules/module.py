import torch
from exceptions import ShapeException


class Module(object):

    def __init__(self, trainable):
        self.trainable = trainable

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def params(self):
        return []

    def set_param(self, name, value):
        pass

    def dim_check(self, tensor_context: str, tensor_: torch.FloatTensor, dim: int):
        # assert tensor_.dim() == 2

        if tensor_.dim() != dim:
            raise ShapeException('Given {} dimension({}), required dimension is {}'
                                 .format(tensor_context, tensor_.dim(), dim))
