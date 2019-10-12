from random import shuffle
from random import seed
from nearest_neighbors import KNNClassifier

import numpy as np


def accuracy_score(y_true, y_pred):
    return 1 - float(np.count_nonzero(y_true - y_pred) / y_true.__len__())


def kfold(n, n_folds,
          stratified=False,
          random_seed=np.nan):
    index_list = []
    for index in range(n):
        index_list.append(index)

    if stratified:
        if random_seed != np.nan:
            seed(random_seed)

        shuffle(index_list)

    result_list = []
    part = 0
    each_len = int(n / n_folds)

    while part < n_folds - 1:
        test_subset = index_list[part * each_len: (part + 1) * each_len]
        train_subset = [x for x in index_list if x not in test_subset]
        result_list.append((train_subset, test_subset))
        part += 1

    test_subset = index_list[part * each_len:]
    train_subset = [x for x in index_list if x not in test_subset]

    result_list.append((train_subset, test_subset))

    return result_list


def knn_cross_val_score(X, Y, k_list, score, cv, **kwargs):

    if X.__len__() != Y.__len__():
        raise TypeError

    if cv is None:
        splits = kfold(X.__len__(), 3)
    else:
        splits = cv

    if score == "accuracy":
        score_func = accuracy_score
    else:
        raise TypeError

    result = {}
    for item in k_list:
        each_acc = np.empty(splits.__len__())

        if kwargs.__len__() != 4:
            raise TypeError

        model = KNNClassifier(item, kwargs["strategy"],
                              kwargs["metric"],
                              kwargs["weights"],
                              kwargs["test_block_size"])

        i = 0
        for train_subset, test_subset in splits:
            tr_x, tr_y = X[train_subset], Y[train_subset]
            te_x, te_y = X[test_subset], Y[test_subset]

            model.fit(tr_x, tr_y)

            res = model.predict(te_x)

            each_acc[i] = score_func(te_y, res)
            i += 1

        result[item] = each_acc

    return result
