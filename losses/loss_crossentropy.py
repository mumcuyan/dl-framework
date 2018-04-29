from . import Loss
import torch


# TODO: LossCrossEntropy'e test yazilacak
class LossCrossEntropy(Loss):

    # not implemented
    def __init__(self, divide_by_n=True, loss_per_row=False):
        super(LossCrossEntropy, self).__init__(divide_by_n, loss_per_row)
        self.input = None

    def forward(self, inputt, target):

        if inputt.dim() == 1:
            inputt = inputt.unsqueeze(0)
        self.input = inputt

        if target.dim() == 1:
            target = target.unsqueeze(-1)
        self.target = target

        if self.loss_per_row:
            output = torch.pow(inputt - target, 2).sum(1)
        else:
            if self.divide_by_n:
                output = torch.pow(inputt - target, 2).sum(1).mean(0)
            else:
                output = torch.pow(inputt - target, 2).sum(1).sum()

        self.loss_logging = (torch.cat((self.loss_logging, output), dim=0))
        return output

    def backward(self):

        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = 2 * (self.input - self.target) * gradwrtoutputt

        if self.divide_by_n:
            dinput /= dinput.shape[0]

        return dinput
