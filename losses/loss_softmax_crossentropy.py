from . import Loss
import torch


class LossSoftmaxCrossEntropy(Loss):

    def __init__(self, divide_by_N=True, loss_per_row=False):
        super(LossSoftmaxCrossEntropy, self).__init__(divide_by_N, loss_per_row)
        self.input = None

    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
    return ce
    """
    def forward(self, y_out: torch.FloatTensor, y_target: torch.FloatTensor):
        self.out, self.target = y_out, y_target

        row_maxs, _ = y_out.max(1)
        x = torch.exp(y_out - row_maxs.repeat(y_out.shape[1], 1).transpose(0, 1))
        tensor_after_softmax = x / x.sum(1).repeat(y_out.shape[1], 1).transpose(0, 1)

        self.probs = tensor_after_softmax
        loss = - torch.log(tensor_after_softmax) * y_target

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
