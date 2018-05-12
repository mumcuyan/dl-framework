from abc import abstractmethod
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

        self.keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        self.train_report = {key: [] for key in self.keys}

    def save_results(self, results: dict, epoch_id, verbose, verbose_freq):

        for key in self.train_report.keys():
            if key in results:
                self.train_report[key].append(results[key])

        if epoch_id % verbose_freq == 0 and verbose == 1:
            self.report_results(results, epoch_id)

    @staticmethod
    def report_results(results: dict, epoch_id):
        print('epoch: {} ---> train_loss: {:.4f}, train_acc: {} ----- val_loss: {:.4f}, val_acc: {}'
                  .format(epoch_id, results['train_loss'], results['train_acc'], results['val_loss'], results['val_acc']))

    @abstractmethod
    def train(self, model, x_train, y_train, batch_size=128, num_of_epoch=100, verbose=0):
        pass
