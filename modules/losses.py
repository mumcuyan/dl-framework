import numpy as np
import torch


class Loss:

    def __init__(self, take_avg, loss_per_row):
        self.out, self.target = None, None

        self.take_avg = take_avg
        self.loss_per_row = loss_per_row
        self.loss_logging = torch.FloatTensor()

    def reset(self):
        self.target = None
        self.out = None

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grad_wrt_output):
        raise NotImplementedError


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

        assert loss is not np.nan
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
