import torch
from . import Loss


class LossMSE(Loss):

    def __init__(self, take_avg=True, loss_per_row=False):
        super(LossMSE, self).__init__(take_avg, loss_per_row)

    def forward(self, y_out: torch.FloatTensor, y_targets: torch.FloatTensor):
        self.out, self.target = y_out, y_targets

        if self.loss_per_row:
            output = torch.pow(self.out - self.target, 2).sum(1)
        else:
            if self.take_avg:
                output = torch.pow(self.out - self.target, 2).sum(1).mean(0)
            else:
                output = torch.pow(self.out - self.target, 2).sum(1).sum()

        self.loss_logging = (torch.cat((self.loss_logging, output), dim=0))

        return output

    def backward(self):
        if self.out is None or self.target is None:
            raise ValueError('Cannot call backward before forward function.\ntarget or input is None')

        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = 2 * (self.out - self.target) * gradwrtoutputt
        self.reset()
        # dinput = torch.sum(dinput, dim=0).unsqueeze(0)
        if self.take_avg:
            dinput /= dinput.shape[0]

        return dinput
