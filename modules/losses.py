import numpy as np
import torch
from .module import Module


class Loss(Module):

    def __init__(self, take_avg: bool):
        super(Loss, self).__init__(trainable=False)
        """
        :param take_avg: flag variable for whether taking avg of data-point each loss
        :param loss_per_row:
        """
        self.out, self.target = None, None
        self.take_avg = take_avg  # or sum over all elements

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

    def __init__(self, take_avg=True):
        super(LossMSE, self).__init__(take_avg)

    def __call__(self, y_out, y_target):
        """
        :param y_out: final
        :param y_target:
        :return:
        """
        loss_val = torch.pow(y_out - y_target, 2).sum(1)
        loss_val = loss_val.mean(0) if self.take_avg else loss_val.sum()

        return loss_val.item()

    def forward(self, y_out: torch.FloatTensor, y_target: torch.FloatTensor):
        self.out = y_out
        self.target = y_target

        return self(y_out, y_target)

    def backward(self):
        """
        :return: derivative of loss function wrt calculated out in forward
        """
        if self.out is None or self.target is None:
            raise ValueError('Cannot call backward before forward function.\ntarget or input is None')

        gradwrtoutputt = torch.FloatTensor([[1]])
        dinput = 2 * (self.out - self.target) * gradwrtoutputt
        self.reset()

        return dinput/dinput.shape[0] if self.take_avg else dinput


class LossCrossEntropy(Loss):

    def __init__(self, take_avg=True):
        super(LossCrossEntropy, self).__init__(take_avg)

    def __call__(self, y_out: torch.FloatTensor, y_target: torch.FloatTensor):
        """
        use this function,
            if the only concern is to calculate loss
            else: use forward to modify out and target attributes (for training)
        :param y_out: tensor.FloatTensor with shape: N x feature_dim
        :param y_target: tensor.FloatTensor with same shape y_out
        :return: value of defined loss function
        """
        
        eps = 1e-6
        y_out.clamp_(min=eps) # set each element to at least eps for numerical stability in log
        log_y_out = torch.log(y_out)
        log_y_out[log_y_out != log_y_out] = 0 # set NaNs to 0
        loss_val = - log_y_out * y_target  # N x 2 dim tensor

        loss_val = loss_val.sum(1)  # loss per row  N x 1 dim tensor
        loss_val = loss_val.mean(0) if self.take_avg else loss_val.sum()
        return loss_val.item()

    def forward(self, y_out: torch.FloatTensor, y_target: torch.FloatTensor):
        """
        :param y_out:
        :param y_target:
        :return:
        """
        # CHECK: given y_out must be a prob distribution e.g: softmax
        row_sum = np.around(y_out.sum(1).numpy()).astype(int)
        ones = np.ones_like(row_sum, dtype=int)
        assert np.array_equal(row_sum, ones)

        self.out = y_out
        self.target = y_target  # save given parameter for backward function
        return self(y_out, y_target)

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
