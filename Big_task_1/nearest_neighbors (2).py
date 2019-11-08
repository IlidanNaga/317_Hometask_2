import numpy as np
import distances
import math
from sklearn.neighbors import NearestNeighbors

def weight(array):
    eps = 0.00001
    return 1 / (array + eps)


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        if strategy != 'my_own':
            if strategy != 'brute' and strategy != 'kd_tree' and strategy != 'ball_tree':
                raise KeyError
            # change parameters
            self.skln_knn = NearestNeighbors(
                n_neighbors=k, algorithm = strategy, metric = metric)
        
    def fit(self, x, y):
        if self.strategy != 'my_own':
            self.skln_knn.fit(x, y)
        else:
            self.train_set = x
        self.train_target = y.astype(int)
    

    def find_kneighbors(self, x, return_distance):
        if self.strategy == 'my_own':
            count_of_blocks = math.ceil(x.shape[0] / self.test_block_size)
            if self.metric == 'euclidean':
                ans_matrix = euclidean_distance(self.train_set, x)
            if self.metric == 'cosine':
                ans_matrix = cosine_distance(self.train_set, x)
            ans_matrix = ans_matrix.T
            neighbour = np.sort(ans_matrix, axis=1)[:, :self.k]
            index = np.argsort(ans_matrix, axis=1)[:,:self.k]
            if return_distance:
                return (neighbour, index)
            return index
        else:
            return self.skln_knn.kneighbors(x, n_neighbors=self.k, return_distance=return_distance)
        
    def predict(self, x):
        answer = np.zeros(x.shape[0])
        blocks = math.ceil(x.shape[0] // self.test_block_size)
        if x.shape[0] <= self.test_block_size:
                blocked_data = x
        else:
            if x.shape[0] % self.test_block_size == 0:
                    blocked_data = np.split(x, blocks)
            else:
                blocks += 1
                array = np.split(x[:(blocks - 1) * self.test_block_size], blocks - 1)
                array.append(x[(blocks - 1) * self.test_block_size:])
                blocked_data = array
        blocked_data = np.array(blocked_data)
        if self.weights:
            for j, block in enumerate(blocked_data):
                dist, ind = self.find_kneighbors(block, True)
                votes = weight(dist)
                for i, mark in enumerate(self.train_target[ind]): # в i-м блоке - i * self.test_block_size ... (i + 1) * -||-
                    ind_array = np.zeros(10)
                    for q in range(self.k):
                        ind_array[mark[q].astype(int)] += votes[i][q]
                    answer[i + j * self.test_block_size] = ind_array.argmax()                   
        
        else:
            for j, block in enumerate(blocked_data):
                ind = self.find_kneighbors(block, False)
                for i, mark in enumerate(self.train_target[ind]):
                    answer[i + j * self.test_block_size] = np.bincount(mark.astype(int)).argmax()
        return answer
