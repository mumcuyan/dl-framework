
class Module(object):

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def set_param(self, name, value):
        pass
