import math
import numpy as np
import torch


class Loss:

    def __init__(self, take_avg: bool, loss_per_row: bool):
        """
        :param take_avg: flag variable for whether taking avg of data-point each loss
        :param loss_per_row:
        """
        self.out, self.target = None, None

        self.take_avg = take_avg
        self.loss_per_row = loss_per_row
        self.loss_logging = []

    def reset(self):
        self.target = None
        self.out = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grad_wrt_output):
        raise NotImplementedError


class LossMSE(Loss):

    def __init__(self, take_avg=True, loss_per_row=False):
        super(LossMSE, self).__init__(take_avg, loss_per_row)

    def __call__(self, y_out, y_target):
        """
        :param y_out: final
        :param y_target:
        :return:
        """
        loss_val = torch.pow(y_out - y_target, 2).sum(1)
        loss_val = loss_val.mean(0) if self.take_avg else loss_val.sum()
        return loss_val

    def forward(self, y_out: torch.FloatTensor, y_target: torch.FloatTensor):

        loss_val = self(y_out, y_target)
        self.out = y_out
        self.target = y_target
        self.loss_logging.append(loss_val)

        return loss_val

    def backward(self):
        """
        :return: derivative of loss function wrt calculated out in forward
        """
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

    def __call__(self, y_out, y_target):
        """
        use this function,
            if the only concern is to calculate loss
            else: use forward to modify out and target attributes (for training)
        :param y_out: tensor.FloatTensor with shape: N x feature_dim
        :param y_target: tensor.FloatTensor with same shape y_out
        :return: value of defined loss function
        """
        eps = pow(np.e, -12)
        loss_val = - torch.log(y_out + eps) * y_target  # N x 2 dim tensor
        loss_val = loss_val.sum(1)  # loss per row  N x 1 dim tensor
        loss_val = loss_val.mean(0) if self.take_avg else loss_val.sum()

        return loss_val[0]  # TODO: handle this accordingly with take_avg false

    def forward(self, y_out: torch.FloatTensor, y_target: torch.FloatTensor):
        """
        :param y_out:
        :param y_target:
        :return:
        """
        # CHECK: given y_out must be a prob distribution e.g: softmax
        assert math.isclose(y_out.sum(), y_out.shape[0], rel_tol=0.01)

        loss = self(y_out, y_target)
        self.out = y_out
        self.target = y_target
        self.loss_logging.append(loss)

        return loss

    # derivation
    # http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
    def backward(self):
        """
        calculate derivative of CrossEntropyLoss wrt input of softmax layer
        so there will be no backward implementation for softmax class
        Please also note that this implementation assumes that softmax object
        is used in final activation function in the network
        :return:
        """
        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = (self.out - self.target) * gradwrtoutputt
        self.reset()

        return dinput/dinput.shape[0] if self.take_avg else dinput
