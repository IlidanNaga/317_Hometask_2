import numpy as np


def replace_nan_to_means(matrix):

    mask_single = np.isnan(matrix)
    mask_column = np.tile(np.all(mask_single, axis=0), (matrix.shape[0], 1))
    mask_no_columns = mask_single * ~mask_column

    res_matrix = np.copy(matrix)
    res_matrix[mask_no_columns] = 0

    sub_matrix = np.tile(res_matrix.sum(axis=0) / (~mask_single).sum(axis=0),
                         (matrix.shape[0], 1))

    sub_matrix[~mask_no_columns] = 0
    res_matrix[mask_column] = 0

    return res_matrix + sub_matrix
