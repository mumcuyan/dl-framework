import torch

class LossSoftmaxCrossEntropy(Loss):

    def __init__(self, target=None, divide_by_N=True, loss_per_row=False):
        super(LossSoftmaxCrossEntropy, self).__init__(target, divide_by_N, loss_per_row)
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

        row_maxs, _ = inputt.max(1)
        x = torch.exp(inputt - row_maxs.repeat(inputt.shape[1], 1).transpose(0, 1))
        tensor_after_softmax = x / x.sum(1).repeat(inputt.shape[1], 1).transpose(0, 1)

        self.probs = tensor_after_softmax
        loss = - torch.log(tensor_after_softmax) * target

        if self.loss_per_row:
            output = loss.sum(1)
        else:
            if self.divide_by_n:
                output = loss.sum(1).mean(0)
            else:
                output = loss.sum(1).sum()

        self.loss_logging = (torch.cat((self.loss_logging, output), dim=0))
        return output

    def backward(self):

        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = (self.probs - self.target) * gradwrtoutputt

        if self.divide_by_n:
            dinput /= dinput.shape[0]

        return dinput
