import torch
from tqdm import trange
from .optimizer import Optimizer
from modules.sequential import Sequential
from collections import OrderedDict
from utils import split_data, batch


class SGD(Optimizer):

    def __init__(self, lr=0.1, momentum_coef=0.0, weight_decay=0.0):
        super(SGD, self).__init__(lr, momentum_coef, weight_decay)

    def train(self, model,
              x_train: torch.FloatTensor,
              y_train: torch.FloatTensor,
              num_of_epochs=100,
              batch_size=128,
              val_split: float = 0.2,
              verbose=0,
              shuffle=True) -> dict:
        """
        :param model: supposed to be a Sequential obj
        :param x_train: torch.FloatTensor
        :param y_train: torch.FloatTensor
        :param batch_size:
        :param verbose: flag parameter for prints
            0 is silent with trange (tqdm)
            1 all results (train_loss, train_acc, val_loss, val_acc)
        :param num_of_epochs:
        :param val_split: float number between 0 and 1 indicating which part of given data will be used for validation
        :param shuffle: boolean flag
        """

        assert x_train.shape[0] == y_train.shape[0]  # sanity check
        if val_split <= 0 or val_split >= 1:
            raise ValueError('validation_split ratio must be between 0 and 1 (exclusive), given {}'
                             .format(val_split))

        (x_train, y_train), (x_val, y_val) = split_data(x_train, y_train, val_split=val_split, is_shuffle=True)
        self._save_gradients(model, is_default=True)

        range_func = trange if verbose == 0 else range
        for i in range_func(num_of_epochs):

            for x_train_batch, y_train_batch in batch(x_train, y_train, batch_size=batch_size):  # go through each batch
                self._update_params(model, x_train_batch, y_train_batch)

            train_acc, train_loss = model.evaluate(x_train, y_train)
            val_acc, val_loss = model.evaluate(x_val, y_val)
            results = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}

            self.save_results(results, i, verbose, verbose_freq=100)

        return self.train_report

    def _update_params(self, model: Sequential, x_train, y_train) -> dict:

        model.forward(x_train, y_train)
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

    def _save_gradients(self, model: Sequential, is_default=False):
        for i, module in enumerate(model.trainable_modules):
            self.log_grad[i] = OrderedDict()

            for name, param in module.params:
                self.log_grad[i][name] = 0 if is_default else module.grads[name]
