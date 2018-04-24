import torch
from .optimizer import Optimizer
from modules.sequential import Sequential
from losses import Loss
from collections import OrderedDict


class SGD(Optimizer):

    def __init__(self, loss: Loss, lr=0.1, momentum_coef=0.0):
        super(SGD, self).__init__(loss, lr, momentum_coef)

    def train(self, model: Sequential, x_train, y_train, num_of_epochs, verbose=0):
        self.save_gradients(model, is_default=True)
        for i in range(num_of_epochs):
            self.update_params(model, x_train, y_train)

    def update_params(self, model: Sequential, x_train: torch.FloatTensor, y_train: torch.FloatTensor):
        y_out = model.forward(x_train)
        self.loss.forward(y_out, y_train)
        model.backward(self.loss.backward())

        for i, module in enumerate(model.trainable_modules):  # go over fully connected layers
            for name, param in module.params:  # go over weight and bias (if not None)
                update = module.grads(name) + self.momentum_coef * self.log_grad[i][name]
                self.log_grad[i][name] = module.grads(name)
                new_param = param - self.lr * update
                module.set_param(name, new_param)

    def save_gradients(self, model: Sequential, is_default=False):
        for i, module in enumerate(model.trainable_modules):
            self.log_grad[i] = OrderedDict()

            for name, param in module.params:
                self.log_grad[i][name] = 0 if is_default else module.grads[name]
