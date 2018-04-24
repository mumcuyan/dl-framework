from .module import Module
import torch


class LossModule(Module):

    def forward(self, *input):
        raise NotImplementedError

    def backward (self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def set_param(self, name, value):
        pass

# TODO: LossCrossEntropy'e test yazilacak
class LossCrossEntropy(LossModule):

    # not implemented
    def __init__(self, target=None, divide_by_N=True, loss_per_row=False, is_trainable=False):
        super(LossCrossEntropy, self).__init__()

        self.divide_by_N = divide_by_N
        self.loss_per_row = loss_per_row
        self.is_trainable = is_trainable

        self.loss_logging = torch.FloatTensor()

        self.input = None
        self.target = target

    def set_options(self, new_divide_by_N, new_loss_per_row):
        self.divide_by_N = new_divide_by_N
        self.loss_per_row = new_loss_per_row

    def forward(self, *input):

        if len(input) >= 2:
            inputt = input[0]
            target = input[1]
        else:
            if len(input) == 1 and self.target is not None:
                inputt = input[0]
                target = self.target
            else:
                raise Exception("At least 2 inputs must be provided")

        if inputt.dim() == 1:
            inputt = inputt.unsqueeze(0)
        self.input = inputt

        if target.dim() == 1:
            target = target.unsqueeze(-1)
        self.target = target

        if self.loss_per_row:
            output = torch.pow(inputt - target, 2).sum(1)
        else:
            if self.divide_by_N:
                output = torch.pow(inputt - target, 2).sum(1).mean(0)
            else:
                output = torch.pow(inputt - target, 2).sum(1).sum()

        self.loss_logging = (torch.cat((self.loss_logging, output), dim=0))
        return output

    def backward(self, *gradwrtoutput):

        if len(gradwrtoutput) == 0:
            gradwrtoutputt = torch.FloatTensor([[1]])
        else:
            if gradwrtoutput[0].dim() == 1:
                gradwrtoutputt = gradwrtoutput[0].unsqueeze(0)
            else:
                gradwrtoutputt = gradwrtoutput[0]

        dinput = 2 * (self.input - self.target) * gradwrtoutputt

        if self.divide_by_N:
            dinput /= dinput.shape[0]

        return dinput


class LossMSE(LossModule):

    def __init__(self, target=None, divide_by_N=True, loss_per_row=False, is_trainable=False):
        super(LossMSE, self).__init__()

        self.divide_by_N = divide_by_N
        self.loss_per_row = loss_per_row
        self.is_trainable = is_trainable

        self.loss_logging = torch.FloatTensor()

        self.input = None
        self.target = target

    def set_options(self, new_divide_by_N, new_loss_per_row):
        self.divide_by_N = new_divide_by_N
        self.loss_per_row = new_loss_per_row

    def forward(self, tensor_in):

        if tensor_in.dim() == 1:
            tensor_in = tensor_in.unsqueeze(0)

        self.input = tensor_in

        if self.loss_per_row:
            output = torch.pow(tensor_in - self.target, 2).sum(1)
        else:
            if self.divide_by_N:
                output = torch.pow(tensor_in - self.target, 2).sum(1).mean(0)
            else:
                output = torch.pow(tensor_in - self.target, 2).sum(1).sum()

        self.loss_logging = (torch.cat((self.loss_logging, output), dim=0))
        return output

    def backward(self, *gradwrtoutput):

        if len(gradwrtoutput) == 0:
            gradwrtoutputt = torch.FloatTensor([[1]])
        else:
            if gradwrtoutput[0].dim() == 1:
                gradwrtoutputt = gradwrtoutput[0].unsqueeze(0)
            else:
                gradwrtoutputt = gradwrtoutput[0]

        dinput = 2 * (self.input - self.target) * gradwrtoutputt

        # dinput = torch.sum(dinput, dim=0).unsqueeze(0)

        if self.divide_by_N:
            dinput /= dinput.shape[0]

        return dinput

    def param(self):
        return []


class LossSoftmaxCrossEntropy(LossModule):

    def __init__(self, target=None, divide_by_N=True, loss_per_row=False, is_trainable=False):
        super(LossSoftmaxCrossEntropy, self).__init__()

        self.divide_by_N = divide_by_N
        self.loss_per_row = loss_per_row
        self.is_trainable = is_trainable

        self.loss_logging = torch.FloatTensor()

        self.input = None
        self.target = target

    def set_options(self, new_divide_by_N, new_loss_per_row):
        self.divide_by_N = new_divide_by_N
        self.loss_per_row = new_loss_per_row

    def forward(self, *input):

        if len(input) >= 2:
            inputt = input[0]
            target = input[1]
        else:
            if len(input) == 1 and self.target is not None:
                inputt = input[0]
                target = self.target
            else:
                raise Exception("At least 2 inputs must be provided")

        if inputt.dim() == 1:
            inputt = inputt.unsqueeze(0)
        self.input = inputt

        if target.dim() == 1:
            target = target.unsqueeze(-1)
        self.target = target

        row_maxs, _ = inputt.max(1)
        x = torch.exp(inputt - row_maxs.repeat(inputt.shape[1], 1).transpose(0, 1))
        tensor_after_softmax = x / x.sum(1).repeat(inputt.shape[1], 1).transpose(0, 1)

        self.probs = tensor_after_softmax
        loss = - torch.log(tensor_after_softmax) * target

        if self.loss_per_row:
            output = loss.sum(1)
        else:
            if self.divide_by_N:
                output = loss.sum(1).mean(0)
            else:
                output = loss.sum(1).sum()

        self.loss_logging = (torch.cat((self.loss_logging, output), dim=0))
        return output

    def backward(self, *gradwrtoutput):

        if len(gradwrtoutput) == 0:
            gradwrtoutputt = torch.FloatTensor([[1]])
        else:
            if gradwrtoutput[0].dim() == 1:
                gradwrtoutputt = gradwrtoutput[0].unsqueeze(0)
            else:
                gradwrtoutputt = gradwrtoutput[0]

        dinput = (self.probs - self.target) * gradwrtoutputt

        if self.divide_by_N:
            dinput /= dinput.shape[0]

        return dinput

    def param(self):
        return []
