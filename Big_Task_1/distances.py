import numpy as np


def euclidean_distance(arr1, arr2):
    res_arr = np.empty((arr1.shape[0], arr2.shape[0]))
    for enum, item in enumerate(arr1[:, ]):
        res_arr[enum] = np.sqrt(np.sum((arr2 - item) ** 2, axis=1))

    return res_arr


def cosine_distance(arr1, arr2):

    res_arr = np.empty((arr1.shape[0], arr2.shape[0]))
    for enum, item in enumerate(arr1[:, ]):
        res_arr[enum] = np.sum(arr2 * item, axis=1) / (
            (np.sqrt(np.sum(arr2 * arr2, axis=1)) *
             np.sqrt((np.sum(item * item)))))

    return 1 - res_arr
