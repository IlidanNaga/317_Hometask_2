from sklearn.neighbors import NearestNeighbors
import numpy as np

from distances import euclidean_distance
from distances import cosine_distance


def weight_function(distance):
    epsilon = 0.00001
    return float(1 / (distance + epsilon))


class KNNClassifier:

    def __init__(self,
                 k,
                 strategy,
                 metric,
                 weights,
                 test_block_size=0):

        self.strategy = strategy

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

        if strategy != "my_own":
            if strategy != "kd_tree" and strategy != "ball_tree" and strategy != "brute":
                raise TypeError

            self.uses = NearestNeighbors(n_neighbors=k,
                                         metric=metric,
                                         algorithm=strategy)

    def fit(self, X, y):
        # my implementation
        if X.__len__() != y.__len__():
            raise TypeError

        self.data = X
        self.target = y
        self.clusters = np.sort(np.unique(y))
        self.clusters_amount = np.unique(y).__len__()
        if self.strategy != "my_own":
            self.uses.fit(X, y)

    def find_kneighbors(self, X, return_distance=True):
        if self.strategy == "my_own":

            ranges = self.metric(self.data, X).T
            nearest = np.argsort(ranges, axis=1)[:, :self.k]

            if return_distance:
                distances = np.empty((X.shape[0], self.k))

                for enum, item in enumerate(nearest):
                    distances[enum] = ranges[enum, nearest[enum]]

                return distances, nearest
            return nearest
        else:
            return self.uses.kneighbors(X, n_neighbors=self.k, return_distance=return_distance)

    def predict(self, X):

        if X.shape[1] != self.data.shape[1]:
            raise TypeError

        test_target = np.empty(X.__len__()).astype(int)

        if self.test_block_size == 0:

            if self.weights:

                distances, nearest = self.find_kneighbors(X, True)

                for enum in range(X.__len__()):
                    cluster_nb = np.zeros(self.clusters_amount)
                    for numb, item in enumerate(nearest[enum]):
                        cluster_nb[np.where(self.clusters ==
                                            self.target[item])[0]] += weight_function(distances[enum, numb])

                    test_target[enum] = self.clusters[np.argmax(cluster_nb)]
            else:
                nearest = self.find_kneighbors(X, False)

                for enum in range(X.__len__()):
                    cluster_nb = np.zeros(self.clusters_amount).astype(int)
                    for numb, item in enumerate(nearest[enum]):
                        cluster_nb[np.where(self.clusters ==
                                            self.target[item])[0]] += 1

                    test_target[enum] = self.clusters[np.argmax(cluster_nb)]

            return test_target
        else:

            if self.weights:

                blocks_am = int(X.__len__() / self.test_block_size)
                for i in range(blocks_am):
                    if i < blocks_am - 1:
                        sub_distances, sub_nearest = self.find_kneighbors(X[i *
                                                                            self.test_block_size:(i+1) *
                                                                            self.test_block_size, :], True)
                    else:
                        sub_distances, sub_nearest = self.find_kneighbors(X[i *
                                                                            self.test_block_size:, :], True)

                    for enum in range(self.test_block_size):
                        cluster_nb = np.zeros(self.clusters_amount)
                        for numb, item in enumerate(sub_nearest[enum]):
                            cluster_nb[np.where(self.clusters ==
                                                self.target[item])[0]] += \
                                weight_function(sub_distances[enum, numb])

                        test_target[enum + i * self.test_block_size] = self.clusters[np.argmax(cluster_nb)]
            else:

                blocks_am = int(X.__len__() / self.test_block_size)

                for i in range(blocks_am):
                    if i < blocks_am - 1:
                        nearest = self.find_kneighbors(X[i *
                                                         self.test_block_size:(i+1) *
                                                         self.test_block_size, :], False)

                    else:
                        nearest = self.find_kneighbors(X[i * self.test_block_size:, :], False)

                    for enum in range(self.test_block_size):
                        cluster_nb = np.zeros(self.clusters_amount).astype(int)
                        for numb, item in enumerate(nearest[enum]):
                            cluster_nb[np.where(self.clusters ==
                                                self.target[item])[0]] += 1

                        test_target[enum + i * self.test_block_size] = self.clusters[np.argmax(cluster_nb)]

            return test_target
