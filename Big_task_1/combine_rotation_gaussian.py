from sklearn.datasets import fetch_mldata
from time import time
from math import sqrt
from skimage.filters import gaussian
from skimage.transform import rotate

from random import seed, shuffle

import numpy as np

from cross_validation import *

# лучший - 30 набор
# это 0.7 фильтрация + 35 угол


mnist = fetch_mldata("MNIST-original")
data = mnist.data / 255.0
target = mnist.target.astype("int0")


# i'd take smaller subsets for faster calculations
seed(1024)
indixes_list = np.arange(70000)
shuffle(indixes_list)
sub_data = data[indixes_list[:3500]]
sub_target = target[indixes_list[:3500]]

cv = kfold(3500, 3)

hold = time()
result_rot_0 = knn_cross_val_score(sub_data,
                                   sub_target,
                                   [4],
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

for filter_size in [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]:
    data_filtrated = np.empty(sub_data.shape)
    for enum in range(sub_data.__len__()):
        data_filtrated[enum] = gaussian(sub_data[enum], sqrt(filter_size), preserve_range=True)

    for rot_angle in [-45, -35, -25, -15, -5, 5, 15, 25, 35, 45]:
        data_combined = np.empty(data_filtrated.shape)
        for enum in range(data_filtrated.__len__()):
            data_combined[enum] = rotate(data_filtrated[enum], rot_angle, preserve_range=True)

        data_combined = data_combined.reshape(3500, 28 * 28)

        result_filter.append(knn_cross_val_score(data_combined,
                                       sub_target,
                                       [4],
                                       "accuracy",
                                       cv,
                                       strategy="my_own",
                                       metric="cosine",
                                       weights=True,
                                       test_block_size=0))

result_filter.append(result_rot_0)

score = np.empty(11 * 10 + 1)

for enum, item in enumerate(result_filter):
    score[enum] = np.sum(item[4])


print(score)
print(np.max(score), np.argmax(score), int(np.argmax(score) / 11), np.argmax(score) % 11)
