from sklearn.datasets import fetch_mldata
from random import seed
from random import shuffle

import numpy as np

from cross_validation import kfold
from cross_validation import knn_cross_val_score

# т.к. косинусная сработала лучше, дальше будет использоваться она

mnist = fetch_mldata("MNIST-original")
data = mnist.data / 255.0
target = mnist.target.astype("int0")

seed(1024)
indixes_list = np.arange(70000)
shuffle(indixes_list)
sub_data = data[indixes_list[:3500]]
sub_target = target[indixes_list[:3500]]

cv = kfold(3500, 3)

result_non_weighted = knn_cross_val_score(sub_data,
                                    sub_target,
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    "accuracy",
                                    cv,
                                    strategy="my_own",
                                    metric="cosine",
                                    weights=False)

result_weighted = knn_cross_val_score(sub_data,
                                    sub_target,
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                    "accuracy",
                                    cv,
                                    strategy="my_own",
                                    metric="cosine",
                                    weights=True,
                                    test_block_size=0)

print("Non-weighted: ", result_non_weighted)
print("Weighted: ", result_weighted)

better = 0

for item in range(1, 11):
    if np.sum(result_non_weighted[item]) > np.sum(result_weighted[item]):
        better += 1
    else:
        better -= 1

if better > 0:
    print("Non-weighted algorithm better")
else:
    print("Weighted algorithm better")
