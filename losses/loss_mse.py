import torch
from . import Loss


class LossMSE(Loss):

    def __init__(self, divide_by_n=True, loss_per_row=False):
        super(LossMSE, self).__init__(divide_by_n, loss_per_row)

    def forward(self, y_out: torch.FloatTensor, y_targets: torch.FloatTensor):
        self.out, self.target = y_out, y_targets

        if self.loss_per_row:
            output = torch.pow(self.out - self.target, 2).sum(1)
        else:
            if self.divide_by_n:
                output = torch.pow(self.out - self.target, 2).sum(1).mean(0)
            else:
                output = torch.pow(self.out - self.target, 2).sum(1).sum()

        self.loss_logging = (torch.cat((self.loss_logging, output), dim=0))
        print("loss: {}".format(output))
        return output

    def backward(self):
        if self.out is None or self.target is None:
            raise ValueError('Cannot call backward before forward function.\ntarget or input is None')

        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = 2 * (self.out - self.target) * gradwrtoutputt
        self.reset()
        # dinput = torch.sum(dinput, dim=0).unsqueeze(0)
        if self.divide_by_n:
            dinput /= dinput.shape[0]

        return dinput
