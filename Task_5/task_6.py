from functools import partial


def update_wrapper(func1, func2):
    func1.__module__ = func2.__module__
    func1.__name__ = func2.__name__
    func1.__doc__ = func2.__doc__
    return func1


def substitutive(f):
    args_am = f.__code__.co_argcount

    def wrapper(*args, **kwargs):

        if args.__len__() == args_am:
            return f(*args, **kwargs)
        elif args.__len__() > args_am:
            raise TypeError

        used_args = []

        if args.__len__() != 0:
            used_args += args

        return update_wrapper(partial(wrapper, *used_args), f)

    return update_wrapper(wrapper, f)
