from nearest_neighbors import KNNClassifier

import numpy as np

from time import sleep

def weight_function(distance):
    epsilon = 0.00001
    return float(1 / (distance + epsilon))


def accuracy_score(x, y):
    return 1 - (np.count_nonzero(x - y) / x.__len__())


def kfold(n, n_folds):

    index_list = np.arange(n)

    result_list = []
    part = 0
    each_len = int(n / n_folds)

    while part < n_folds - 1:
        test_subset = np.array(index_list[part * each_len: (part + 1) * each_len])
        train_subset = np.array([x for x in index_list if x not in test_subset])
        result_list.append((train_subset, test_subset))
        part += 1

    test_subset = np.array(index_list[part * each_len:])
    train_subset = np.array([x for x in index_list if x not in test_subset])

    result_list.append((train_subset, test_subset))

    return result_list


def knn_cross_val_score(X, Y, k_list, score, cv, **kwargs):

    if X.__len__() != Y.__len__():
        raise TypeError

    if cv is None:
        cv = kfold(X.__len__(), 3)

    if score == "accuracy":
        score_func = accuracy_score
    else:
        raise TypeError

    max_k = max(k_list)

    result = {}
    each_acc = np.empty([len(k_list), len(cv)])

    for enumer, fold in enumerate(cv):

        trX, trY = X[fold[0]], Y[fold[0]]
        teX, teY = X[fold[1]], Y[fold[1]]

        model = KNNClassifier(k=max_k, **kwargs)
        model.fit(trX, trY)
        clusters = np.sort(np.unique(trY))
        clusters_amount = np.unique(trY).__len__()

        distances, nearest = model.find_kneighbors(teX)

        test_target = np.empty(teX.__len__()).astype(int)

        for ite, it in enumerate(k_list):

            new_distances = distances[:, :it]
            new_nearest = nearest[:, :it]

            if "weights" in kwargs.keys():
                if kwargs["weights"]:
                    for enum in range(teX.__len__()):
                        cluster_nb = np.zeros(clusters_amount)
                        for numb in range(new_nearest.shape[1]):
                            cluster_nb[trY[new_nearest[enum, numb]]] += weight_function(new_distances[enum, numb])

                        test_target[enum] = clusters[np.argmax(cluster_nb)]
                else:
                    for enum in range(teX.__len__()):
                        cluster_nb = np.zeros(clusters_amount)
                        for numb in range(new_nearest.shape[1]):
                            cluster_nb[trY[new_nearest[enum, numb]]] += 1

                        test_target[enum] = clusters[np.argmax(cluster_nb)]
            else:

                for enum in range(teX.__len__()):
                    cluster_nb = np.zeros(clusters_amount)
                    for numb in range(new_nearest.shape[1]):
                        cluster_nb[trY[new_nearest[enum, numb]]] += 1

                    test_target[enum] = clusters[np.argmax(cluster_nb)]

            each_acc[ite, enumer] = score_func(teY, test_target)

    for i in range(len(k_list)):
        result[k_list[i]] = each_acc[i]

    return result
