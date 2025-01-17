import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.square(x).sum(axis=1)[:, None] + np.square(y).sum(axis=1) - 2 * np.dot(x, y.T))


def cosine_distance(x, y):
    return 1 - np.dot(x, y.T) / np.sqrt(np.sum(x ** 2, axis=1)[:, None]) / np.sqrt(np.sum(y ** 2, axis=1)[None, :])
