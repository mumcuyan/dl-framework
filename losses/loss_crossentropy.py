from losses import Loss
import torch

# TODO: LossCrossEntropy'e test yazilacak
class LossCrossEntropy(Loss):

    # not implemented
    def __init__(self, target=None, divide_by_N=True, loss_per_row=False):
        super(LossCrossEntropy, self).__init__(target, divide_by_N, loss_per_row)
        self.input = None

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
