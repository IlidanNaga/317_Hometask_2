from cross_validation import knn_cross_val_score
from cross_validation import kfold

from sklearn.datasets import fetch_mldata
from random import seed
from random import shuffle

import numpy as np

# part_2

mnist = fetch_mldata("MNIST-original")
data = mnist.data / 255.0
target = mnist.target.astype("int0")

 # последний параметр - это число тестовых переменных, для которых разом вычисляются соседи

# i'd take smaller subsets for faster calculations
seed(1024)
indixes_list = np.arange(70000)
shuffle(indixes_list)
sub_data = data[indixes_list[:3500]]
sub_target = target[indixes_list[:3500]]


cv = kfold(3500, 3, False)

result_euclidean = knn_cross_val_score(sub_data,
                                       sub_target,
                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                       "accuracy",
                                       cv,
                                       strategy="brute",
                                       metric="euclidean",
                                       weights=False,
                                       test_block_size=0)

result_cosine = knn_cross_val_score(sub_data,
                                    sub_target,
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    "accuracy",
                                    cv,
                                    strategy="my_own",
                                    metric="cosine",
                                    weights=False,
                                    test_block_size=0)

print("Euclidean results: ", result_euclidean)
print("Cosine results: ", result_cosine)
better = 0

for item in range(1, 11):
    if np.sum(result_euclidean[item]) > np.sum(result_cosine[item]):
        better += 1
    else:
        better -= 1

if better > 0:
    print("Euclidean worked better")
else:
    print("Cosine worked better")
