from modules.layers import Linear

def static_vars(**kwargs):
    def real_decorator(func):
        def decorate(*arg1):
            print("inside decorator: {}".format(func))
            for k in kwargs:
                setattr(func, k, kwargs[k])
            return func(*arg1)
        return decorate
    return real_decorator


def static_vars_2(**kwargs):
    def real_decorator(func):
        for k in kwargs:
            print(func, " -- ", k, " -- ", kwargs[k])
            setattr(func, k, kwargs[k])
        return func()
    return real_decorator


@static_vars_2(var=12)
def basic_func():
    print(basic_func)
    # basic_func.var = 12
    print("here: {}".format(basic_func))
    print(basic_func.var)
    #print("arg1: {} -- arg2: {}".format(arg1, arg2))

k = basic_func()
print("func_result: {}".format(k))


class MyClass:

    def __init__(self):
        self._aras = 22

    @property
    def aras(self):
        return self._aras

    @static_vars(var=12)
    def basic_func(self, my_var):
        if not hasattr(self.basic_func, 'var'):
            print("no attribute !")
            self.basic_func.var = 12
        print("here: {}".format(self.basic_func))
        print(self.basic_func.var)

"""
obj = MyClass()
obj.basic_func(22)
res = hasattr(obj, 'aras')
print("Res: {}".format(res))

"""

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        print(func)
        return func
    return decorate


@static_vars(counter=0)
def foo(arg):
    print(foo, " -- ", arg)
    foo.counter += 1
    print("Counter is %d" % foo.counter)

# foo(arg=2)
# basic_func(1, 2)


lin = Linear(out=12, input_size=6, activation='relu')
if hasattr(lin, 'input_size'):
    print("Hey there !!!! ")