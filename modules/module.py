import torch
from exceptions import ShapeException


class Module(object):

    def __init__(self, trainable, name=None):
        self._trainable = trainable
        self._name = name

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def params(self):
        return []

    def set_param(self, name, value):
        pass

    @property
    def trainable(self):
        return self._trainable

    @property
    def name(self):
        return self._name

    def dim_check(self, tensor_context: str, tensor_: torch.FloatTensor, dim: int):
        # assert tensor_.dim() == 2

        if tensor_.dim() != dim:
            raise ShapeException('Given {} dimension({}), required dimension is {}'
                                 .format(tensor_context, tensor_.dim(), dim))
