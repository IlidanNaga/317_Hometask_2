import numpy as np


def get_max_before_zero(vector):

    target = np.where(vector == 0)[0] + 1
    if target[-1] >= vector.shape[0]:
        target = target[:-1]

    if target.shape[0] == 0:
        return None

    return vector[target].max()
