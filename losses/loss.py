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
