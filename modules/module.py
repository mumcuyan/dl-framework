from exceptions import ShapeException


def require_dimension(dim):
    def decorator(f):
        def inner_func(self, tensor):
            if tensor.dim() != dim:
                raise ShapeException('func: dimension({}), required dimension is {}, given {}'
                                     .format(f.__name__, tensor.dim(), dim, tensor.dim()))
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

    @property
    def params(self):
        keys = []
        for key in keys:
            if self._params[key] is not None:
                yield key, self._params[key]

    def set_param(self, name, value):
        pass
    
    @property
    def trainable(self):
        return self._trainable

    @property
    def name(self):
        return self._name
