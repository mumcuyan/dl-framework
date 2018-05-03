from exceptions import ShapeException


def require_train(f):
    def wrapper(self, *args):
        if self.model is None:
            raise Exception("Model is not trained")
        return f(self, *args)

    return wrapper


def require_dimension(dim):
    def decorator(f):
        def inner_func(self, tensor):
            if tensor.dim() != dim:
                raise ShapeException('func: dimension({}), required dimension is {}'
                                     .format(f.__name__, tensor.dim(), dim))
            return f(self, tensor)
        return inner_func
    return decorator


def require_not_none(attr_name):
    def decorator(f):
        def inner_func(self, *args):
            if getattr(self, attr_name, None) is None:
                raise ValueError('Attribute called {} cannot be None, '
                                 'forward function should be called before taking gradient !'.format(attr_name))
            return f(self, *args)
        return inner_func
    return decorator


class Module(object):

    def __init__(self, trainable, name=None):
        self._trainable = trainable
        self._name = name

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def params(self):
        return []

    def set_param(self, name, value):
        pass

    @property
    def trainable(self):
        return self._trainable

    @property
    def name(self):
        return self._name
