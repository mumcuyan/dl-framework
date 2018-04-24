import torch


class Loss:

    def __init__(self, divide_by_n, loss_per_row):
        self.out, self.target = None, None

        self.divide_by_n = divide_by_n
        self.loss_per_row = loss_per_row
        self.loss_logging = torch.FloatTensor()

    def reset(self):
        self.target = None
        self.out = None

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grad_wrt_output):
        raise NotImplementedError
