import numpy as np
import skimage

from sklearn.datasets import fetch_mldata
from random import shuffle
from random import seed
from time import time

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

result_rot = []
time_rot = []

sub_data = sub_data.reshape(sub_data.__len__(), 28, 28)

for rot_angle in [-15, -10, 5, 5, 10, 15]:
    sub_data = sub_data.reshape(sub_data.__len__(), 28, 28)
    rotated_data = np.empty(sub_data.shape)
    for item in range(sub_data.__len__()):
        rotated_data[item] = skimage.transform.rotate(sub_data[item], rot_angle, preserve_range=True)
    rotated_data = rotated_data.reshape(rotated_data.__len__(), 28 * 28)
    print(rotated_data.shape)
    hold = time()
    result_rot.append(knn_cross_val_score(rotated_data,
                                          sub_target,
                                          [2, 3, 4, 5, 7],
                                          "accuracy",
                                          cv,
                                          strategy="my_own",
                                          metric="cosine",
                                          weights=True,
                                          test_block_size=0))
    time_rot.append(time() - hold)

print("Without rotation: ", result_rot_0)
print("With rotation -15: ", result_rot[0])
print("With rotation -10: ", result_rot[1])
print("With rotation -5: ", result_rot[2])
print("With rotation 5: ", result_rot[3])
print("With rotation 10: ", result_rot[4])
print("With rotation 15: ", result_rot[5])

# сдвиг на -10 градусов, кажется, работает лучше всего... надо посмотреть

result_rot.append(result_rot_0)
score = np.empty((7, 5))

for enum, item in enumerate(result_rot):
    score[enum, 0] = np.sum(item[2])
    score[enum, 1] = np.sum(item[3])
    score[enum, 2] = np.sum(item[4])
    score[enum, 3] = np.sum(item[5])
    score[enum, 4] = np.sum(item[7])

print(score)
print(np.max(score))