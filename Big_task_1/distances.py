import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(np.sum(x ** 2, axis=1)[:, np.newaxis] + np.sum(y ** 2, axis=1) - 2 * np.dot(x, y.T))


def cosine_distance(x, y):
    return 1 - np.dot(x, y.T).T / np.sqrt((x ** 2).sum(axis=1)) / np.sqrt((y ** 2).sum(axis=1))[:, None]
