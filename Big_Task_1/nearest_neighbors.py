from distances import euclidean_distance
from distances import cosine_distance

from sklearn.neighbors import KNeighborsClassifier as KNC
import numpy as np


def weight_function(distance):
    epsilon = 0.00001
    return float(1 / (distance + epsilon))


def array_weight_function(distances):
    epsilon = 0.00001
    result = np.empty(distances.shape)

    for x in range(distances.shape[0]):
        for y in range(distances.shape[1]):
            result[x, y] = float(1 / distances[x, y] + epsilon)

    return result


class KNNClassifier:

    def __init__(self,
                 k,
                 strategy,
                 metric,
                 weights,
                 test_block_size=0):

        self.strategy = strategy

        if strategy == "my_own":
            # my implementation
            self.k = k
            self.weights = weights
            self.data = None
            self.target = None
            self.clusters = None
            self.clusters_amount = None
            self.test_block_size = test_block_size

            if metric == "euclidean":
                self.metric = euclidean_distance
            elif metric == "cosine":
                self.metric = cosine_distance
            else:
                raise TypeError

        elif strategy == "brute":
            print("brute")
            if weights:
                self.uses = KNC(k,
                                algorithm='brute',
                                weights=weight_function,
                                n_jobs=-1)
            else:
                self.uses = KNC(k,
                                algorithm='brute',
                                weights='uniform',
                                n_jobs=-1)
        elif strategy == 'kd_tree':
            if weights:
                self.uses = KNC(k,
                                algorithm='kd_tree',
                                weights=array_weight_function,
                                n_jobs=-1)
            else:
                self.uses = KNC(k,
                                algorithm='kd_tree',
                                weights='uniform',
                                n_jobs=-1)
        elif strategy == 'ball_tree':
            if weights:
                self.uses = KNC(k,
                                algorithm='ball_tree',
                                weights=weight_function,
                                n_jobs=-1)
            else:
                self.uses = KNC(k,
                                algorithm='ball_tree',
                                weights='uniform',
                                n_jobs=-1)
        else:
            raise TypeError

    def fit(self, X, y):
        if self.strategy == "my_own":
            # my implementation
            if X.__len__() != y.__len__():
                raise TypeError

            self.data = X
            self.target = y
            self.clusters = np.sort(np.unique(y))
            self.clusters_amount = np.unique(y).__len__()
        else:
            self.uses.fit(X, y)

    def find_kneighbors(self, X, return_distance):
        if self.strategy == "my_own":

            ranges = self.metric(self.data, X).T
            nearest = np.argpartition(ranges, self.k)[:, :self.k]

            if return_distance:
                distances = np.empty((X.shape[0], self.k))

                for enum, item in enumerate(nearest):
                    distances[enum] = ranges[enum, nearest[enum]]

                return nearest, distances
            return nearest
        else:
            return self.uses.kneighbors(X, return_distance)

    def predict(self, X):

        if self.strategy == "my_own":

            if X.shape[1] != self.data.shape[1]:
                raise TypeError

            test_target = np.empty(X.__len__()).astype(int)

            if self.weights:
                nearest, distances = self.find_kneighbors(X, True)

                for enum in range(X.__len__()):
                    cluster_nb = np.zeros(self.clusters_amount)
                    for numb, item in enumerate(nearest[enum]):
                        cluster_nb[np.where(self.clusters ==
                                            self.target[item])[0]] += \
                            weight_function(distances[enum, numb])

                    test_target[enum] = \
                        self.clusters[self.clusters[np.argmax(cluster_nb)]]
            else:
                nearest = self.find_kneighbors(X, False)

                for enum in range(X.__len__()):
                    cluster_nb = np.zeros(self.clusters_amount).astype(int)
                    for numb, item in enumerate(nearest[enum]):
                        cluster_nb[np.where(self.clusters ==
                                            self.target[item])[0]] += 1

                    test_target[enum] = \
                        self.clusters[self.clusters[np.argmax(cluster_nb)]]

            return test_target

        else:
            return self.uses.predict(X)
