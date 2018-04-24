from collections import OrderedDict
from modules.sequential import Sequential


class SGD:

    def __init__(self, loss_func, lr=0.1, momentum_coef=0.0):
        super(SGD, self).__init__(loss_func)
        self.lr = lr
        self.momentum_coef = momentum_coef
        self.log_grad = OrderedDict()

    # TODO do verbose
    def train(self, model, input, num_of_epochs=100, verbose=0):
        self.save_gradients(model, is_default=True)
        for i in range(num_of_epochs):
            self.update_params(input, model)

    def update_params(self, input, model: Sequential):

        model.forward(input)
        model.backward()

        for i, module in enumerate(model.trainable_modules):  # go over fully connected layers
            for name, param in module.params:  # go over weight and bias (if not None)
                update = module.grads[name] + self.momentum_coef * self.log_grad[i][name]
                self.log_grad[i][name] = module.grads[name]
                new_param = param - self.lr * update
                module.set_param(name, new_param)

    def save_gradients(self, model: Sequential, is_default=False):
        for i, module in enumerate(model.trainable_modules):
            self.log_grad[i] = OrderedDict()

            for name, param in module.params:
                self.log_grad[i][name] = 0 if is_default else module.grads[name]
