from tqdm import trange
import torch
from .optimizer import Optimizer
from modules.sequential import Sequential
from collections import OrderedDict


class SGD(Optimizer):

    def __init__(self, lr=0.1, momentum_coef=0.0, weight_decay=0.0):
        super(SGD, self).__init__(lr, momentum_coef, weight_decay)

    def train(self, model: Sequential, x_train, y_train, num_of_epochs, verbose=0):
        self.save_gradients(model, is_default=True)
        for i in trange(num_of_epochs):
            self.update_params(model, x_train, y_train)

    def update_params(self, model: Sequential, x_train: torch.FloatTensor, y_train: torch.FloatTensor):
        _, loss = model.forward(x_train, y_train)
        model.backward()

        for i, module in enumerate(model.trainable_modules):  # go over fully connected layers
            for name, param in module.params:  # go over weight and bias (if not None)
                update = module.grads[name]

                if self.weight_decay > 0:
                    update += self.weight_decay * module.grads[name]

                if self.momentum_coef > 0:
                    update += self.momentum_coef * self.log_grad[i][name]
                    self.log_grad[i][name] = module.grads[name]

                new_param = param - self.lr * update
                module.set_param(name, new_param)

    def save_gradients(self, model: Sequential, is_default=False):
        for i, module in enumerate(model.trainable_modules):
            self.log_grad[i] = OrderedDict()

            for name, param in module.params:
                self.log_grad[i][name] = 0 if is_default else module.grads[name]
