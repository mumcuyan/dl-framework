from collections import OrderedDict
from .sequential import Sequential
# TODO: create a base class


class Optimizer:
    pass


class SGD(Optimizer):

    def __init__(self, lr=0.1, momentum_coef=0.0):
        self.lr = lr
        self.momentum_coef = momentum_coef
        self.log_grad = OrderedDict()

    def train(self, input, sequential, num_of_epochs=100):
        self.save_gradients(sequential, is_default=True)
        for i in range(num_of_epochs):
            self.update_params(input, sequential)

    def update_params(self, input, sequential: Sequential):

        sequential.forward(input)
        sequential.backward()

        for i, module in enumerate(sequential.modules.values()):
            if module.is_trainable:
                for name, param in module.params.items():
                    if param is not None:
                        update = module.grads[name] + self.momentum_coef * self.log_grad[i][name]
                        self.log_grad[i][name] = module.grads[name]
                        new_param = param - self.lr * update
                        module.set_param(name, new_param)

    def save_gradients(self, sequential, is_default=False):
        for i, module in enumerate(sequential.modules.values()):
            self.log_grad[i] = OrderedDict()
            if module.is_trainable:
                for name, param in module.params.items():
                    if param is not None:
                        if is_default:
                            self.log_grad[i][name] = 0
                        else:
                            self.log_grad[i][name] = module.grads[name]
