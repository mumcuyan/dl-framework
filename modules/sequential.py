from .losses import Loss, LossMSE, LossCrossEntropy
from .activations import *
from .layers import Linear, Dropout

from exceptions import InputSizeNotFoundError, NotCompatibleError
from utils import one_hot2label
from collections import OrderedDict

import collections
import json
import numpy as np
import torch


class Sequential(Module):

    def __init__(self, modules=None, loss_func=None):
        """
        :param modules: must behave like Sequence which can be either list or tuple (built-in)
        :param loss_func: is a Loss defined in losses.py which will be called for loss calculation
        """
        super(Sequential, self).__init__(trainable=False)
        
        # modules
        self._modules = OrderedDict()
        self.layer_sizes = []
        if modules is not None:
            if isinstance(modules, collections.Sequence):
                for idx, module in enumerate(modules):
                    self.add(module)
            else:
                raise TypeError('Given parameter type {} is invalid, required: {} '
                                .format(type(modules), "collections.Sequence"))
        
        # loss function
        self._loss = loss_func

    def add(self, module: Module):
        """
        this function can be used for adding layers (Activations, Linear or Dropout)
        :param module: is a Module (not Sequential)
        """

        if module is None or isinstance(module, Loss):
            raise ValueError('Given object type is Loss is not valid !')
        if module is None or not isinstance(module, Module):
            raise ValueError('Given object type {} is not Module '.format(type(module)))

        if hasattr(module, 'input_size'):
            if module.input_size is None and len(self.layer_sizes) == 0:
                raise InputSizeNotFoundError('First module must specify input size in Linear layer !')
            if module.input_size is not None and \
                    (len(self.layer_sizes) != 0 and module.input_size != self.layer_sizes[-1]):
                raise NotCompatibleError('Given input size {} is not compatible with previous layer size: {}'
                                 .format(module.input_size,  self.layer_sizes[-1]))

            if module.input_size is None:
                module.input_size = self.layer_sizes[-1]

            module.initialize()
            self.layer_sizes.append(getattr(module, 'output_size'))
        
        # if checks are OK, add module to Sequential with default name
        default_name = str(len(self._modules)) + "_" + module.__class__.__name__
        module.name = default_name
        self._modules[default_name] = module

        try:
            self.add(getattr(module, 'activation'))
        except AttributeError:
            pass

    def forward(self, x_input: torch.FloatTensor, y_input: torch.FloatTensor=None):
        """
        This function will be used for training, for more detail you may refer to Optimizer's train method to
        see how it is used.

        :param x_input: input data is passed, must be 2D
        :param y_input: if this parameter passed, this function is used for training, if not it is test or prediction
        :return: predicted values for given dataset (x_input) based on network
        """
        
        # check if train or test
        train = y_input is not None
        y_pred = x_input
        for module in self._modules.values():
            # handle train and test for Dropout diffrently, if test no forward for Dropout is needed
            if isinstance(module, Dropout) and not train:
                continue
            # take forward pass for each module in Sequential
            y_pred = module.forward(y_pred)

        if train:  # calculate loss for training
            self._loss.forward(y_pred, y_input)

        return y_pred

    def backward(self):
        """
        start backward propogation in reverse order, each module takes gradient wrt to output
        """
        
        # gradient of loss wrt output of model
        gradwrtoutputt = self._loss.backward()
        # take backward pass for each module in Sequential, gradients are stored in attributes of each module
        for module in reversed(list(self._modules.values())):
            gradwrtoutputt = module.backward(gradwrtoutputt)

    def predict(self, x_test: torch.FloatTensor, convert_label=False):
        """
        :param x_test:
        :param convert_label: boolean flag whether to convert prob dist to label
        :return: return N x 2 size tensor as a y_prediction
        """
        if not torch.is_tensor(x_test):
            raise TypeError('Given x_test parameter must be torch.Tensor !')
        
        # make prediction on given data
        y_pred = self.forward(x_test)
        return y_pred if not convert_label else y_pred.max(1)[1]

    def evaluate(self, x_test: torch.FloatTensor, y_test: torch.FloatTensor, return_pred=False):
        """
        To get accuracy, loss and prediction of x_test based on trained model, use this function.
        If you do not have correct answers please refer to .predict function defined for this object.

        :param x_test: [N x feature_dim] torch.FloatTensor
        :param y_test: [N x class_num] torch.FloatTensor (one-hot encoded)
        :param return_pred: flag var for returning labels for each test data point --> [N x 1] torch.FloatTensor
        :return: accuracy of prediction and loss value, based on return_pred, predictions as labels
        """
        if not torch.is_tensor(y_test):
            raise TypeError('Given x_test parameter must be torch.Tensor !')
        
        # make prediction on given data
        y_pred = self.predict(x_test)
        # calculate loss for prediction
        loss_val = self._loss(y_pred, y_test)
        # calculate accuracy for prediction
        acc_val = self.accuracy(one_hot2label(y_pred), one_hot2label(y_test))

        if not return_pred:
            return acc_val, loss_val
        else:
            return acc_val, loss_val, y_pred.max(1)[1]

    def save_to_disk(self, filename, is_save_to_disk=True):
        """
        Save model to disk in JSON file format with given filename
        
        :param filename: str, filename
        :param is_save_to_disk: flag variable for save to disk or return params of each module as OrderedDict
        :return: params of each module as OrderedDict
        """
        
        # initialize dump
        dump = OrderedDict()
        # initialize all_params
        all_params = OrderedDict()
        # traverse each module in Sequential
        for name_module, module in self._modules.items():
            # temporary OrderedDict
            param_dict = OrderedDict()
            # traverse each param of module
            for name_param, param in module.params:  # go over weight and bias (if not None)
                # add them to temporary OrderedDict
                param_dict[name_param] = param.data.numpy().tolist()
            # if module is of type Dropout, handle differently
            if isinstance(module, Dropout):
                # save probability of dropout to dict as well
                param_dict["p_dropout"] = 1 - module.prob
            # store OrderedDictionary of params of module in all_params
            all_params[name_module] = param_dict
        
        # save loss function for model
        dump["loss"] = 'MSE' if isinstance(self.loss, LossMSE) else 'CE'
        # save all params of all modules
        dump["modules"] = all_params
        
        
        # if save to disk
        if is_save_to_disk:
            # save as JSON file
            with open(filename, 'w') as f:
                json.dump(dump, f) 
        # return dictionary of all params
        else:
            return all_params

    @classmethod
    def load_from_disk(cls, filename):
        """
        Load model from disk in JSON file format with given filename
        
        :param filename: str, filename
        :return: Sequential, model that is loaded
        """
        
        # load JSON file
        with open(filename) as f:
            dump = json.load(f)
        
        # take loss
        loss = dump["loss"]
        # take all params
        all_params = dump["modules"]
        # initialize model
        model = cls()
        
        # traverse dictionary of all parameters
        for name_module, module_params in all_params.items():
            # check names and find type of modules
            char_index = name_module.find('_')
            module_type = name_module[char_index+1:]
            
            # handle differently according to type of module
            if module_type == "Linear":
                casted_param = torch.from_numpy(np.array(module_params['weight'])).type(torch.FloatTensor)
                module = Linear(out=casted_param.shape[1], input_size=casted_param.shape[0])
            elif module_type == "Dropout":
                module = Dropout(prob=module_params["p_dropout"])
            elif module_type == "ReLU":
                module = ReLU()
            elif module_type == "Tanh":
                module = Tanh()
            elif module_type == "Softmax":
                module = Softmax()
            else:
                raise ValueError('Given module_type {} is not valid !'.format(module_type))
            
            # add recovered module to model
            model.add(module)
            
            if module_type == "Linear":
                for name_param, param in module_params.items():  # go over weight and bias (if not None)
                    casted_param = torch.from_numpy(np.array(param)).type(torch.FloatTensor)
                    model._modules[name_module].set_param(name_param, casted_param)
        
        # add loss to model
        model.loss = LossMSE() if loss == 'MSE' else LossCrossEntropy()

    def __eq__(self, other):
        """
        equality check for each module as well as loss function ?
        :param other: Sequential
        :return: boolean value whether both object are the same structure or not
        """
        for (module, other_module) in (self.modules, other.modules):
            if module != other_module:
                return False

        # it does not compare loss attribute
        return True

    def __str__(self):
        """
        string representation of each module is accumulated in a string
        to print summary of the model
        :return: string -> summary of model
        """
        model_summary = "*" * 62 + "\n"
        model_summary += "Layer Name \t\t Input Shape\t\t Output Shape\n"
        model_summary += "*" * 62 + "\n"
        for module in self.modules:
            model_summary += str(module) + "\n"

        model_summary += "*" * 62 + "\n"
        model_summary += 'Loss Function: \t\t{} \n'.format(self.loss.__class__.__name__)

        model_summary += "*" * 62 + "\n"
        return model_summary

    @staticmethod
    def accuracy(y_pred, y_test):
        """
        given predictions and true values, calculates percentile of success
        :return float value as a percentages
        """
        if not torch.is_tensor(y_test):
            raise TypeError('Given x_test parameter must be torch.Tensor !')
        
        # return accuracy for given predicted target and actual targets
        return (y_pred == y_test).type(torch.FloatTensor).mean().item()

    def print_model(self):
        print(self)

    @property
    def module_names(self):
        return list(self._modules.keys())

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss_func):
        # set loss function
        self._loss = loss_func

    @property
    def modules(self):
        for module in self._modules.values():
            yield module

    @property
    def trainable_modules(self):
        """
        trainable attribute used for getting
        :return:
        """
        # generator for trainable modules
        for module in self._modules.values():
            if module.trainable:
                yield module

