
class ShapeException(Exception):
    """
    Some functions take some specific shape of input data
    if this constraint could not be met, ShapeException is raised
    """
    pass


class InputSizeNotFoundError(Exception):
    """
    Input size of first layer must be specified explicitly

    """
    pass


class NotCompatibleError(Exception):
    """
    Output units of previous layer must match input of current layer.
    If this requirement could not be met, NotCompatibleError is raised
    """
    pass


class ValidationSetNotFound(Exception):
    """
    If neither validation set nor validation split is passed to train method for
    Optimizer
    """

    pass