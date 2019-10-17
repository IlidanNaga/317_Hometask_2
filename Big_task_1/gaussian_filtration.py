from sklearn.datasets import fetch_mldata
from time import time
from math import sqrt
from skimage.filters import gaussian

from random import seed, shuffle

import numpy as np

from cross_validation import *


mnist = fetch_mldata("MNIST-original")
data = mnist.data / 255.0
target = mnist.target.astype("int0")


# i'd take smaller subsets for faster calculations
seed(1024)
indixes_list = np.arange(70000)
shuffle(indixes_list)
sub_data = data[indixes_list[:3500]]
sub_target = target[indixes_list[:3500]]

cv = kfold(3500, 5)

hold = time()
result_rot_0 = knn_cross_val_score(sub_data,
                                   sub_target,
                                   [2, 3, 4, 5, 7],
                                   "accuracy",
                                   cv,
                                   strategy="my_own",
                                   metric="cosine",
                                   weights=True,
                                   test_block_size=0)
time_rot_0 = time() - hold

result_filter = []
time_filter = []

sub_data = sub_data.reshape(3500, 28, 28)

for filter_size in [0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5]:
    data_filtrated = np.empty(sub_data.shape)
    for enum in range(sub_data.__len__()):
        data_filtrated[enum] = gaussian(sub_data[enum], sqrt(filter_size), preserve_range=True)

    data_filtrated = data_filtrated.reshape(3500, 28 * 28)

    result_filter.append(knn_cross_val_score(data_filtrated,
                                   sub_target,
                                   [2, 3, 4, 5, 7],
                                   "accuracy",
                                   cv,
                                   strategy="my_own",
                                   metric="cosine",
                                   weights=True,
                                   test_block_size=0))

print("With filtration 0.5: ", result_filter[0])
print("With filtration 0.7: ", result_filter[1])
print("With filtration 0.9: ", result_filter[2])
print("With filtration 1: ", result_filter[3])
print("With filtration 1.1: ", result_filter[4])
print("With filtration 1.3: ", result_filter[5])
print("With filtration 1.5: ", result_filter[6])
print("Without filtration: ", result_rot_0)

result_filter.append(result_rot_0)

score = np.empty((8, 5))

for enum, item in enumerate(result_filter):
    score[enum, 0] = np.sum(item[2])
    score[enum, 1] = np.sum(item[3])
    score[enum, 2] = np.sum(item[4])
    score[enum, 3] = np.sum(item[5])
    score[enum, 4] = np.sum(item[7])

print(score)
print(np.max(score))
