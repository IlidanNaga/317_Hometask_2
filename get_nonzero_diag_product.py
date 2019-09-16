import numpy as np


def get_nonzero_diag_product(matrix):

    diag = np.diag(matrix)

    if np.all(diag == 0):
        return None
    else:
        return np.prod(diag[diag != 0])
