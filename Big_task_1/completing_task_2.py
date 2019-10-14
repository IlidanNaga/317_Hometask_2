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

# part_1

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
                                       strategy="my_own",
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
    result_better = knn_cross_val_score(sub_data,
                                        sub_target,
                                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        "accuracy",
                                        cv,
                                        strategy="my_own",
                                        metric="euclidean",
                                        weights=True,
                                        test_block_size=0)
else:
    print("Cosine worked better")
    result_better = knn_cross_val_score(sub_data,
                                        sub_target,
                                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                        "accuracy",
                                        cv,
                                        strategy="my_own",
                                        metric="cosine",
                                        weights=True,
                                        test_block_size=0)

print("Weighted for better performed method: ", result_better)

# Euclidean results:  {1: array([0.89879931, 0.9219554 , 0.90839041]), 2: array([0.88336192, 0.89622642, 0.89811644]), 3: array([0.90137221, 0.91595197, 0.89982877]), 4: array([0.90566038, 0.91423671, 0.90325342]), 5: array([0.90737564, 0.91166381, 0.89982877]), 6: array([0.90051458, 0.90994854, 0.89726027]), 7: array([0.90566038, 0.90909091, 0.9015411 ]), 8: array([0.90051458, 0.90909091, 0.8989726 ]), 9: array([0.90051458, 0.90651801, 0.89726027]), 10: array([0.89365352, 0.90823328, 0.89383562])}
# Cosine results:  {1: array([0.9193825 , 0.9348199 , 0.93835616]), 2: array([0.91080617, 0.92624357, 0.92465753]), 3: array([0.91166381, 0.93138937, 0.92465753]), 4: array([0.91423671, 0.93310463, 0.92893836]), 5: array([0.91509434, 0.93138937, 0.91695205]), 6: array([0.90909091, 0.93396226, 0.91438356]), 7: array([0.91595197, 0.93396226, 0.90839041]), 8: array([0.90737564, 0.93138937, 0.91010274]), 9: array([0.91509434, 0.9245283 , 0.90839041]), 10: array([0.90909091, 0.92109777, 0.91523973])}
