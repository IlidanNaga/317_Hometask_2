import numpy as np
import math
from nearest_neighbors import KNNClassifier

def accuracy(prediction, answer):
    return len(np.where(prediction == answer)[0]) / len(prediction)

def kfold(n, n_folds):
    if n_folds > n or n_folds < 2:
        raise TypeError
    
    ind_array = np.arange(n)
    if n % n_folds == 0:
        ind_array = np.split(ind_array, n_folds)
    else:
        array = np.split(ind_array[:(n % n_folds) * (n // n_folds + 1)], n % n_folds)
        array = array + np.split(ind_array[(n % n_folds) * (n // n_folds + 1):], n_folds - n % n_folds)
        ind_array = array                             
    answer = []
    for elem in ind_array:
        dop = np.concatenate([np.arange(elem[0]), np.arange(min(elem[::-1][0] + 1, n), n)])
        answer.append((dop, elem))
    return answer
def knn_cross_val_score(x, y, k_list, score, cv, **kwargs):
    if cv == None:
        cv = kfold(x.shape[0], 3)
    answer = {}
    dop_answer = np.zeros(len(cv) * len(k_list)).reshape(len(k_list), -1)
    model = KNNClassifier(k=max(k_list), **kwargs)
    for j, data in enumerate(cv):
        train_set = x[data[0]]
        train_target = y[data[0]]
        test_set = x[data[1]]
        test_target = y[data[1]]
        model.fit(train_set, train_target)
        if kwargs['weights']:
            dist, ind = model.find_kneighbors(test_set, return_distance=True)
            votes = 1 / (dist + 0.00001)
        else:
            ind = model.find_kneighbors(test_set, return_distance=False)
        for i, k in enumerate(k_list):
            sub_answer = np.zeros(test_set.shape[0])
            if kwargs['weights']:
                k_votes = votes[:, :k]
            k_ind = ind[:, :k]
            if kwargs['weights']:
                for q in range(len(test_set)):
                    ind_array = np.zeros(10)
                    for num in range(k_ind.shape[1]):
                        ind_array[train_target[k_ind[q][num]].astype(int)] += k_votes[q][num]
                    sub_answer[q] = np.argmax(ind_array)
            else:
                for q in range(len(test_set)):
                    ind_array = np.zeros(10)
                    for num in range(k_ind.shape[1]):
                        ind_array[train_target[k_ind[q][num]].astype(int)] += 1
                    sub_answer[q] = np.argmax(ind_array)

            dop_answer[i][j] = 1 - len(np.where(test_target != sub_answer)[0]) / len(test_target)
    for i, k in enumerate(k_list):
        answer[k] = dop_answer[i]
    return answer
