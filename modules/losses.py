import numpy as np
import torch


class Loss:

    def __init__(self, take_avg, loss_per_row):
        """
        :param take_avg:
        :param loss_per_row:
        """
        self.out, self.target = None, None

        self.take_avg = take_avg
        self.loss_per_row = loss_per_row
        # self.loss_logging = torch.FloatTensor()
        self.loss_logging = []

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

        #if self.loss_per_row:
        #     output = torch.pow(self.out - self.target, 2).sum(1)
        # else:

        loss = torch.pow(self.out - self.target, 2).sum(1)
        loss = loss.mean(0) if self.take_avg else loss.sum()

        self.loss_logging = (torch.cat((self.loss_logging, loss), dim=0))
        return loss

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


class LossCrossEntropy(Loss):

    def __init__(self, take_avg=True, loss_per_row=False):
        super(LossCrossEntropy, self).__init__(take_avg, loss_per_row)

    @staticmethod
    def apply_softmax(y_linear: torch.FloatTensor):
        row_maxs, _ = y_linear.max(1)
        x = torch.exp(y_linear - row_maxs.repeat(y_linear.shape[1], 1).transpose(0, 1))
        return x / x.sum(1).repeat(y_linear.shape[1], 1).transpose(0, 1)

    def forward(self, y_linear: torch.FloatTensor, y_target: torch.FloatTensor, val):
        print("@loss_forward: -> {}".format(y_linear.shape))

        y_out = self.apply_softmax(y_linear) # y_out is softmax distribution of y_linear
        if y_target is None:
            return y_out

        eps = pow(np.e, -12)
        loss = - torch.log(y_out + eps) * y_target  # N x 2 dim tensor
        loss = loss.sum(1)  # loss per row  N x 1 dim tensor

        loss = loss.mean(0) if self.take_avg else loss.sum()
        if val is False:
            self.out = y_out
            self.target = y_target
            self.loss_logging.append(loss)

        # self.loss_logging = torch.cat((self.loss_logging, loss_out), dim=0)
        return loss

    # derivation
    # http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
    def backward(self):
        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = (self.out - self.target) * gradwrtoutputt

        self.reset()
        if self.take_avg:
            dinput /= dinput.shape[0]

        return dinput
