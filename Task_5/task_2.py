from collections.abc import Iterable


def linearize(args):
    fin_list = []

    def add_iterable(cur_item):

        if isinstance(cur_item, Iterable):
            for value in cur_item:
                if isinstance(value, Iterable):
                    if type(value) == str:
                        if len(value) == 1:
                            fin_list.append(value)
                            continue

                    add_iterable(value)
                else:
                    fin_list.append(value)

        else:
            fin_list.append(cur_item)

    for item in args:
        add_iterable(item)

    return fin_list
