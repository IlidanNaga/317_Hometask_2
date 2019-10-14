import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from random import shuffle
from random import seed
from time import time

from nearest_neighbors import KNNClassifier
from cross_validation import accuracy_score as ac_s


mnist = fetch_mldata("MNIST-original")
data = mnist.data / 255.0
target = mnist.target.astype("int0")

 # последний параметр - это число тестовых переменных, для которых разом вычисляются соседи

# part_1

# i'd take smaller subsets for faster calculations
seed(1024)
indixes_list = np.arange(70000)
shuffle(indixes_list)
sub_data = data[indixes_list[:3500]]
sub_target = target[indixes_list[:3500]]


# that's for sub_split split, 3000/500
trX, teX, trY, teY = train_test_split(sub_data, sub_target, test_size=1/7)

"""List of all algorithms:
1) my_own, euclidian
2) kd_tree
3) brute
4) ball_tree
"""

# selecting 10/20/100 objects
all_descriptors = np.arange(28*28)

seed(666)

shuffle(all_descriptors)
selected_10 = np.copy(all_descriptors[:10])

shuffle(all_descriptors)
selected_20 = np.copy(all_descriptors[:20])

shuffle(all_descriptors)
selected_100 = np.copy(all_descriptors[:100])

print(trX[:, selected_10].shape)
print(trY.shape)

print("10 selected descriptors are {}\n20 - {}\n100 - {}".format(selected_10,
                                                                 selected_20,
                                                                 selected_100))

model = KNNClassifier(5, "my_own", "euclidean", False)
model.fit(trX[:, selected_10], trY)
result_10 = model.predict(teX[:, selected_10])

model.fit(trX[:, selected_20], trY)
result_20 = model.predict(teX[:, selected_20])

model.fit(trX[:, selected_100], trY)
result_100 = model.predict(teX[:, selected_100])

print("Accuracy for: \n10 - {}\n20 - {}\n100 - {}".format(ac_s(teY, result_10),
                                                          ac_s(teY, result_20),
                                                          ac_s(teY, result_100)))

time_hold = time()
model.fit(trX, trY)
result_my_own = model.predict(teX)
time_my_own = time() - time_hold

model = KNNClassifier(5, "my_own", "cosine", False)
time_hold = time()
model.fit(trX, trY)
result_my_cosine = model.predict(teX)
time_my_cosine = time() - time_hold

model = KNNClassifier(5, "my_own", "euclidean", True)
time_hold = time()
model.fit(trX, trY)
result_my_weighted_euc = model.predict(teX)
time_my_weighted_euc = time() - time_hold

model = KNNClassifier(5, "my_own", "cosine", True)
time_hold = time()
model.fit(trX, trY)
result_my_weighted_cos = model.predict(teX)
time_my_weighted_cos = time() - time_hold

model = KNNClassifier(5, "kd_tree", "euclidean", False)
time_hold = time()
model.fit(trX, trY)
result_kd_tree = model.predict(teX)
time_kd_tree = time() - time_hold

model = KNNClassifier(5, "kd_tree", "euclidean", True)
time_hold = time()
model.fit(trX, trY)
result_kd_weight = model.predict(teX)
time_kd_weight = time() - time_hold

model = KNNClassifier(5, "brute", "euclidean", False)
time_hold = time()
model.fit(trX, trY)
result_brute = model.predict(teX)
time_brute = time() - time_hold

model = KNNClassifier(5, "brute", "euclidean", True)
time_hold = time()
model.fit(trX, trY)
result_brute_weight = model.predict(teX)
time_brute_weight = time() - time_hold

model = KNNClassifier(5, "ball_tree", "euclidean", False)
time_hold = time()
model.fit(trX, trY)
result_ball_tree = model.predict(teX)
time_ball_tree = time() - time_hold

model = KNNClassifier(5, "ball_tree", "euclidean", True)
time_hold = time()
model.fit(trX, trY)
result_ball_weight = model.predict(teX)
time_ball_weight = time() - time_hold

print("my_own: accuracy = {}, time = {}".format(ac_s(teY, result_my_own), time_my_own))
print("my cosine: accuracy = {}, time = {}".format(ac_s(teY, result_my_cosine), time_my_cosine))
print("my weight euc: accuracy = {}, time = {}".format(ac_s(teY, result_my_weighted_euc), time_my_weighted_euc))
print("my weight cos: accuracy = {}, time = {}".format(ac_s(teY, result_my_weighted_cos), time_my_weighted_cos))
print("kd_tree: accuracy = {}, time = {}".format(ac_s(teY, result_kd_tree), time_kd_tree))
print("kd_weight: accuracy = {}, time = {}".format(ac_s(teY, result_kd_weight), time_kd_weight))
print("brute: accuracy = {}, time = {}".format(ac_s(teY, result_brute), time_brute))
print("brute_weight: accuracy = {}, time = {}".format(ac_s(teY, result_brute_weight), time_brute_weight))
print("ball_tree: accuracy = {}, time = {}".format(ac_s(teY, result_ball_tree), time_ball_tree))
print("ball_weight: accuracy = {}, time = {}".format(ac_s(teY, result_ball_weight), time_ball_weight))