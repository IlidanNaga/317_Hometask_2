import numpy as np


def encode_rle(vector):

    positions = np.flatnonzero(~np.isclose(vector[:-1], vector[1:])) + 1
    positions = np.append(0, positions)

    lengths = np.diff(np.append(positions, len(vector)))

    return vector[positions], lengths
