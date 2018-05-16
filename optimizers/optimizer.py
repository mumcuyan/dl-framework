from abc import abstractmethod
from collections import OrderedDict
import logging


class Optimizer:

    def __init__(self, lr=0.1, momentum_coef=0, weight_decay=0.0):
        """
        Abstract class for Optimizers, each concrete class must extends this function overwriting train method

        :param lr: learning rate
        :param momentum_coef: momentum coefficient
        :param weight_decay: lambda as a L2 regularization parameter
        """

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

        self.keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        self.train_report = {key: [] for key in self.keys}

    def save_results(self, results: dict, epoch_id, verbose, verbose_freq):
        """
        after each epoch this function is called
        :param results: is a form of dictionary having {train_loss, train_acc, val_loss, val_acc}
        :param epoch_id: current epoch num to report performance of model to stdout
        :param verbose: flag whether report performance of model to stdout
        :param verbose_freq: frequency of
        :return:
        """
        for key in self.train_report.keys():
            if key in results:
                self.train_report[key].append(results[key])

        if epoch_id != 0 and epoch_id % verbose_freq == 0 and verbose == 1:
            self.report_results(results, epoch_id)

    @staticmethod
    def report_results(results: dict, epoch_id):
        """
        This function prints performance of model to stdout, with a frequency of verbose_freq passed to save_results
        :param results: dict of {train_loss, train_acc, val_loss, val_acc}
        :param epoch_id: current epoch id to report
        """
        print('epoch: {} ---> train_loss: {:.4f}, train_acc: {:.4f} ----- val_loss: {:.4f}, val_acc: {:.4f}'
                  .format(epoch_id, results['train_loss'], results['train_acc'], results['val_loss'], results['val_acc']))

    @abstractmethod
    def train(self, model, x_train, y_train, batch_size=128, num_of_epoch=100, verbose=0) -> dict:
        """
        This is an abstract method for all classes
        :param model: is Sequential object having list of Modules
        :param x_train: 2D data N x #feature
        :param y_train: 2D data N x #class
        :param batch_size: batch size of gradient update
        :param num_of_epoch: epoch number
        :param verbose: flag variable 0 quiet, 1 verbose
        :return: train report having list of {train_loss, train_acc, val_loss, val_acc} for each epoch
        """
        raise NotImplementedError
