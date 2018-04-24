from collections import OrderedDict
from .module import Module
from .losses import LossModule
import torch

# TODO: make self.modules private create a generator for optimizer .train()


class Sequential(Module):

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = OrderedDict()
        self.loss_func = None
        self.output = None
        self.output_before_loss = None

    def set_loss_function(self, ):
        pass

    def add_module(self, module, name=None):

        # TODO: after removing loss module, remove or LossModule instance
        # and
        #  if module is None or (not isinstance(module, Module) and not isinstance(module, LossModule)):
        #    raise ValueError('Given object type {} is not Module '.format(type(module)))

        if name is None:
            name = len(self.modules)

        self.modules[name] = module

    def forward(self, input: torch.FloatTensor):

        tmp_input = input
        if input.dim() == 1:
            tmp_input = tmp_input.unsqueeze(0)

        for module in self.modules.values():
            if isinstance(module, LossModule):
                self.output_before_loss = tmp_input
            tmp_input = module.forward(tmp_input)

        self.output = tmp_input
        if self.output_before_loss is None:
            self.output_before_loss = self.output

        # TODO ?
        _, self.prediction = self.output_before_loss.max(1)

        return tmp_input

    # TODO: delete this part
    def forward_without_loss(self, input: torch.FloatTensor):

        tmp_input = input
        if input.dim() == 1:
            tmp_input = tmp_input.unsqueeze(0)

        for module in self.modules.values():
            if isinstance(module, LossModule):
                self.output_before_loss = tmp_input
            else:
                tmp_input = module.forward(tmp_input)

        self.output = tmp_input
        if self.output_before_loss is None:
            self.output_before_loss = self.output
        _, self.prediction = self.output_before_loss.max(1)

        return tmp_input

    def backward(self, *gradwrtoutput):

        if len(gradwrtoutput) > 0:
            raise ValueError("Backward @Sequential does not exists")

        gradwrtoutputt = torch.FloatTensor([[1]])
        for module in reversed(list(self.modules.values())):
            gradwrtoutputt = module.backward(gradwrtoutputt)

        self.output = None
        self.prediction = None

        return gradwrtoutputt

    def predict(self, input, is_force=True):
        if is_force or self.prediction is None:
            if is_force or self.output is None:
                self.forward_without_loss(input)
            else:
                _, self.prediction = self.output.max(1)

        return self.prediction

    def accuracy(self, input, target, is_force=True):
        self.predict(input, is_force)
        return (self.prediction == target).type(torch.FloatTensor).mean(), self.prediction

    def print_model(self):
        for module in self.modules:
            print(module)


"""
def backward():

if len(gradwrtoutput) == 0:
    gradwrtoutputt = torch.FloatTensor([[1]])
else:
    if gradwrtoutput[0].dim() == 1:
        gradwrtoutputt = gradwrtoutput[0].unsqueeze(0)
    else:
        gradwrtoutputt = gradwrtoutput[0]


:param gradwrtoutput: 
:return: 
"""