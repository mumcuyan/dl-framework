
class Module(object):

    def __init__(self, trainable):
        self.trainable = trainable

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def set_param(self, name, value):
        pass
