from nearest_neighbors import KNNClassifier
from cross_validation import kfold
from cross_validation import knn_cross_val_score

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
from random import shuffle

mnist = fetch_mldata("MNIST-original")

data = mnist.data / 255.0
target = mnist.target.astype("int0")

indexes = []
for item in range(70000):
    indexes.append(item)

shuffle(indexes)

test_data = data[indexes[:2000]]
test_target = target[indexes[:2000]]

print(np.unique(test_target))


cv = kfold(test_data.__len__(), 3)


result = knn_cross_val_score(test_data, test_target, [2, 4], "accuracy", cv, strategy="my_own", metric="euclidean", weights=True, test_block_size=0)

print(result)