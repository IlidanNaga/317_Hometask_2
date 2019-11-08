import numpy as np
import skimage

from sklearn.datasets import fetch_mldata

from cross_validation import *

from random import seed, shuffle

# в целом, даёт минимальные изменения, отработало лучше всего для -3, 3 сдвига


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

result_0 = knn_cross_val_score(sub_data,
                                   sub_target,
                                   [4],
                                   "accuracy",
                                   cv,
                                   strategy="my_own",
                                   metric="cosine",
                                   weights=True,
                                   test_block_size=0)

sub_data = sub_data.reshape(3500, 28, 28)
results = []

for s_1 in [-3, -2, -1, 1, 2, 3]:
    for s_2 in [-3, -2, -1, 1, 2, 3]:
        shifted_1 = np.empty(sub_data.shape)
        if s_1 < 0:
            shifted_1[:, -1 * s_1:, :] = sub_data[:, :s_1, :]
        else:
            shifted_1[:, :-1 * s_1, :] = sub_data[:, s_1:, :]

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

print(results)
print(result_0)

results.append(result_0)

scores = np.empty(6 * 6 + 1)

for enum, item in enumerate(results):
    scores[enum] = np.sum(item[4])


print(scores)
print(np.max(scores), np.argmax(scores), int(np.argmax(scores) / 6), np.argmax(scores) % 6)
