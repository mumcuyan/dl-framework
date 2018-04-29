from . import Loss
import torch


class LossSoftmaxCrossEntropy(Loss):

    def __init__(self, take_avg=True, loss_per_row=False):
        super(LossSoftmaxCrossEntropy, self).__init__(take_avg, loss_per_row)
        self.input = None

    def forward(self, y_linear: torch.FloatTensor, y_target: torch.FloatTensor):
        self.target = y_target

        # print(y_linear)
        row_maxs, _ = y_linear.max(1)
        x = torch.exp(y_linear - row_maxs.repeat(y_linear.shape[1], 1).transpose(0, 1))
        y_out = x / x.sum(1).repeat(y_linear.shape[1], 1).transpose(0, 1)
        self.out = y_out
        # print("TensorOut: {}".format(y_out))

        loss = - torch.log(y_out) * y_target  # N x 2 dim tensor
        loss_out = loss.sum(1)  # loss per row  N x 1 dim tensor

        loss_out = loss_out.mean(0) if self.take_avg else loss_out.sum()
        self.loss_logging = (torch.cat((self.loss_logging, loss_out), dim=0))

        return loss_out

    # derivation
    # http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
    def backward(self):

        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = (self.out - self.target) * gradwrtoutputt

        self.reset()
        if self.take_avg:
            dinput /= dinput.shape[0]

        return dinput
