from collections import OrderedDict
import logging


class Optimizer:
    def __init__(self, lr=0.1, momentum_coef=0):
        self.log_grad = OrderedDict()
        self.lr = lr
        self.momentum_coef = momentum_coef
        self.logger = logging.basicConfig(filename="sample.log", level=logging.INFO)
