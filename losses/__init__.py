import torch
from .loss_mse import LossMSE
from .loss_crossentropy import LossCrossEntropy
from .loss_softmax_crossentropy import LossSoftmaxCrossEntropy

__all__ = ['LossMSE', 'LossCrossEntropy', 'LossSoftmaxCrossEntropy']


class Loss:

    def __init__(self, target, divide_by_n, loss_per_row):
        self.target = target
        self.divide_by_n = divide_by_n
        self.loss_per_row = loss_per_row
        self.loss_logging = torch.FloatTensor()

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grad_wrt_output):
        raise NotImplementedError
