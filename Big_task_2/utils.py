import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """

    res = np.zeros(w.__len__()).astype(np.float64)

    for i in range(w.__len__()):
        w[i] += eps
        res[i] = function(w)
        w[i] -= eps

    return (res - function(w)) / eps

