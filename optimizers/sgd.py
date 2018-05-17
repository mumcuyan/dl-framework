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
        
        # split data as training and validation
        (x_train, y_train), (x_val, y_val) = split_data(x_train, y_train, val_split=val_split, is_shuffle=True)
        self._save_gradients(model, is_default=True)

        range_func = trange if verbose == 0 else range
        for i in range_func(num_of_epochs):

            for x_train_batch, y_train_batch in batch(x_train, y_train, batch_size=batch_size):  # go through each batch
                # update parameters using current batch
                self._update_params(model, x_train_batch, y_train_batch)

            # After epoch is done, evaluate performance of the model simply calling evaluation on
            # both train dataset as well as validation dataset
            train_acc, train_loss = model.evaluate(x_train, y_train)
            val_acc, val_loss = model.evaluate(x_val, y_val)
            results = {'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}

            self.save_results(results, i, verbose, verbose_freq=100)

        return self.train_report

    def _update_params(self, model: Sequential, x_train, y_train):
        """
        after training one-pass by calling forward and backward
        gradients are set as sepertate attribute inside each trainable layers
        this gradient might be modified with weight_decay momentum_coefficient settings
        and set afterward to each corresponding trainable modules

        :param model: is a Sequential object (neural network)
        :param x_train: batch size of x_train vals
        :param y_train: batch size of corresponding y_train vals
        """
        
        # forward pass of model
        model.forward(x_train, y_train)
        # backward pass of model
        model.backward()
        # now each module has log of gradient of each of its parameters

        for i, module in enumerate(model.trainable_modules):  # go over trainable modules (linear layers)
            for name, param in module.params:  # go over weight and bias (if not None)
                # take corresponding gradient as update
                update = module.grads[name]
                
                if self.weight_decay > 0:  # L2 regularization for weights by weight decaying
                    # some fraction of parameter is added to update
                    update += self.weight_decay * module.grads[name]

                if self.momentum_coef > 0:
                    # add fraction of logged gradient from previous update to update, as momentum
                    update += self.momentum_coef * self.log_grad[i][name]
                    # log the gradient for later use, for momentum
                    self.log_grad[i][name] = module.grads[name]
                
                # find new value of parameter
                new_param = param - self.lr * update
                # set module's corresponding parameter to new value
                module.set_param(name, new_param)

    def _save_gradients(self, model: Sequential, is_default=False):
        """
        :param model: is a Sequential object
        :param is_default:
        :return:
        """
        
        # traverse each trainable module
        for i, module in enumerate(model.trainable_modules):
            # initialize log for gradients as OrderedDict
            self.log_grad[i] = OrderedDict()
            
            # traverse each param for module
            for name, param in module.params:
                # log gradient
                self.log_grad[i][name] = 0 if is_default else module.grads[name]
