from exceptions import ShapeException


def require_dimension(dim):
    """
    :param dim: requirement for dimension of the data
    :raise: ShapeException, if tensor as an input does not have the dim as requirement
    """
    def decorator(f):
        def inner_func(self, tensor):
            if tensor.dim() != dim:
                raise ShapeException('func: dimension({}), required dimension is {}, given {}'
                                     .format(f.__name__, tensor.dim(), dim, tensor.dim()))
            return f(self, tensor)
        return inner_func
    return decorator


def require_not_none(attr_name):
    """
    :param attr_name: attribute name of the object that must not be None (used as a cache)
    :raise ValueError if it is none
    """
    def decorator(f):
        def inner_func(self, *args):
            if getattr(self, attr_name, None) is None:
                raise ValueError('Attribute called {} cannot be None, '
                                 'forward function should be called before taking gradient !'.format(attr_name))
            return f(self, *args)
        return inner_func
    return decorator


class Module(object):
    """
    Main abstract class for this module.
    Activations, Loss, Linear Layer are Module and extends this class mainly implementing
    forward and backward methods for training as well as predictions
    """

    def __init__(self, trainable, name=None):
        """
        :param trainable: boolean flag of whether the Module have trainable parameters or not
        :param name: each Module is associated with a unique name that is assigned in Sequential(when constructing model)
        """
        self._trainable = trainable
        self._name = name

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def __str__(self):
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

    @name.setter
    def name(self, value):
        self._name = value
