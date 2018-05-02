from collections import OrderedDict
import logging


class Optimizer:
    def __init__(self, lr=0.1, momentum_coef=0, weight_decay=0.0):

        if weight_decay < 0:
            raise ValueError('Weight decay cannot be negative !')
        if lr < 0:
            raise ValueError('Learning rate cannot be negative !')
        if momentum_coef < 0:
            raise ValueError('Momentum coefficient cannot be negative !')

        self.log_grad = OrderedDict()
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum_coef = momentum_coef
        self.logger = logging.basicConfig(filename="sample.log", level=logging.INFO)

    def train(self, model, x_train, y_train, num_of_epoch, verbose=0):
        raise NotImplementedError
