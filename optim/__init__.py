from .sgd import SGD

__all__ = ['SGD']


class Optimizer:
    def __init__(self, loss_func):
        self.loss_func = loss_func
