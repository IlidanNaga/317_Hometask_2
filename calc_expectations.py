import numpy as np


def calc_expectations(h, w, values, chance):

    sub_matric = np.zeros([chance.shape[0] + h - 1, chance.shape[1] + w - 1])
    sub_matric[h - 1:, w - 1:] += chance

    sub_shape = (h, w)
    view_shape = tuple(np.subtract(sub_matric.shape,
                                   sub_shape) + 1) + sub_shape
    sub_arrays = np.lib.stride_tricks.as_strided(sub_matric,
                                                 view_shape,
                                                 sub_matric.strides * 2)
    sub_arrays = sub_arrays.reshape((-1,) + sub_shape)

    result = sub_arrays.sum(axis=1).sum(axis=1).reshape(chance.shape)

    return result * values
