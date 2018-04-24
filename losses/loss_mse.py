import torch
from losses import Loss


class LossMSE(Loss):

    def __init__(self, target=None, divide_by_N=True, loss_per_row=False):
        super(LossMSE, self).__init__(target, divide_by_N, loss_per_row)
        self.input = None

    def forward(self, tensor_in):

        if tensor_in.dim() == 1:
            tensor_in = tensor_in.unsqueeze(0)

        self.input = tensor_in

        if self.loss_per_row:
            output = torch.pow(tensor_in - self.target, 2).sum(1)
        else:
            if self.divide_by_n:
                output = torch.pow(tensor_in - self.target, 2).sum(1).mean(0)
            else:
                output = torch.pow(tensor_in - self.target, 2).sum(1).sum()

        self.loss_logging = (torch.cat((self.loss_logging, output), dim=0))
        return output

    def backward(self):

        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = 2 * (self.input - self.target) * gradwrtoutputt

        # dinput = torch.sum(dinput, dim=0).unsqueeze(0)

        if self.divide_by_n:
            dinput /= dinput.shape[0]

        return dinput

