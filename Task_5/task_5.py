import functools


def check_arguments(*types):
    def my_decorator(f):
        @functools.wraps(f)
        def wrapper_function(*args):
            if types is not None:
                if args.__len__() < types.__len__():
                    raise TypeError

                for pos in range(types.__len__()):
                    if not isinstance(args[pos], types[pos]):
                        raise TypeError
            return f(*args)

        return wrapper_function

    return my_decorator