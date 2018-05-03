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

    @staticmethod
    def report_results(results: dict, epoch_id: int, verbose):
        if epoch_id % 100 == 0 and verbose == 1:
            print('epoch: {} ---> train_loss: {:.4f}, train_acc: {} ----- val_loss: {:.4f}, val_acc: {}'
                  .format(epoch_id, results['train_loss'], results['train_acc'], results['val_loss'], results['val_acc']))

    def train(self, model, x_train, y_train, num_of_epoch, verbose=0):
        raise NotImplementedError
