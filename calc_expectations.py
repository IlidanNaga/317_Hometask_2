import numpy as np


def calc_expectations(h, w, values, chance):
    # рекурсия запрещена не была :)
    def first_rec(matrix, left):
        current = np.roll(chance, left, axis=0)
        current[0:left] = 0
        matrix += current
        if left + 1 < h:
            matrix = first_rec(matrix, left + 1)

        return matrix

    result_lines = first_rec(np.copy(chance), 1)
    crutch = np.copy(result_lines)

    def second_rec(matrix, left):
        current = np.roll(crutch, left, axis=1)
        current.T[0:left] = 0
        matrix += current
        if left + 1 < w:
            matrix = second_rec(matrix, left + 1)

        return matrix

    result_total = second_rec(result_lines, 1)

    return result_total * values
