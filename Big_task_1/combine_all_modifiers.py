from sklearn.datasets import fetch_mldata
from time import time
from math import sqrt
from skimage.filters import gaussian
from skimage.transform import rotate

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

results = []

sub_data = sub_data.reshape(3500, 28, 28)

i = 0

for filter_size in [0.6, 0.7, 0.8, 1, 1.2, 1.3, 1.4]:
    data_filtrated = np.empty(sub_data.shape)
    for enum in range(sub_data.__len__()):
        data_filtrated[enum] = gaussian(sub_data[enum], sqrt(filter_size), preserve_range=True)

    for rot_angle in [-45, -35, -15, 15, 35, 45]:
        data_combined = np.empty(data_filtrated.shape)
        for enum in range(data_filtrated.__len__()):
            data_combined[enum] = rotate(data_filtrated[enum], rot_angle, preserve_range=True)

        for s_1 in [-3, -2, -1, 1, 2, 3]:
            for s_2 in [-3, -2, -1, 1, 2, 3]:
                shifted_1 = np.empty(data_combined.shape)
                if s_1 < 0:
                    shifted_1[:, -1 * s_1:, :] = data_combined[:, :s_1, :]
                else:
                    shifted_1[:, :-1 * s_1, :] = data_combined[:, s_1:, :]

                shifted_2 = np.empty(sub_data.shape)
                if s_2 < 0:
                    shifted_2[:, :, -1 * s_2:] = shifted_1[:, :, :s_2]
                else:
                    shifted_2[:, :, :-1 * s_2] = shifted_1[:, :, s_2:]

                shifted_2 = shifted_2.reshape(3500, 28 * 28)

                results.append(knn_cross_val_score(shifted_2,
                                                   sub_target,
                                                   [4],
                                                   "accuracy",
                                                   cv,
                                                   strategy="my_own",
                                                   metric="cosine",
                                                   weights=True,
                                                   test_block_size=0))

results.append(result_rot_0)

score = np.empty(7 * 6 * 6 * 6 + 1)

for enum, item in enumerate(results):
    score[enum] = np.sum(item[4])


print(score)
print(np.max(score), np.argmax(score))
