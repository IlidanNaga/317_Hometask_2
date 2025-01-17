from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as ac_s

from nearest_neighbors import KNNClassifier

import numpy as np
from skimage.transform import rotate
from skimage.filters import gaussian
from math import sqrt

# so, the best algorithm is weighed cosine one
# with 4 folds
# with rotate on -5 angle
# with gaussian 1.1 filter

mnist = fetch_mldata("MNIST-original")
data = mnist.data / 255.0
target = mnist.target.astype("int0")

left_matrix = np.zeros((28, 28)).astype(int)
right_matrix = np.zeros((28, 28)).astype(int)

for enum, item in enumerate(left_matrix):
    if enum + 1 < 28:
        item[enum + 1] = 1

for enum, item in enumerate(right_matrix):
    if enum - 2 >= 0:
        item[enum - 2] = 1

trX, teX, trY, teY = train_test_split(data, target, test_size=1/7, random_state=666)

teX = teX.reshape(teX.__len__(), 28, 28)
trX = trX.reshape(trX.__len__(), 28, 28)

teX_rotated = np.empty(teX.shape)
trX_rotated = np.empty(trX.shape)

trX_filtred = np.empty(trX.shape)
teX_filtred = np.empty(teX.shape)

trX_s_1 = np.empty(trX.shape)
teX_s_1 = np.empty(teX.shape)

trX_s_2 = np.empty(trX.shape)
teX_s_2 = np.empty(teX.shape)

for enum in range(teX.__len__()):
    teX_rotated[enum] = rotate(teX[enum], -35, preserve_range=True)
    teX_filtred[enum] = gaussian(teX_rotated[enum], sqrt(0.8), preserve_range=True)

    for i in range(10000):
        teX_s_1[i] = np.dot(left_matrix, teX_filtred[i])
        teX_s_2[i] = np.dot(teX_s_2[i], right_matrix)

print("Modified_teX")

for enum in range(trX.__len__()):
    trX_rotated[enum] = rotate(trX[enum], -35, preserve_range=True)
    trX_filtred[enum] = gaussian(trX_rotated[enum], sqrt(0.8), preserve_range=True)

    for i in range(60000):
        trX_s_1[i] = np.dot(left_matrix, trX_filtred[i])
        trX_s_2[i] = np.dot(trX_s_2[i], right_matrix)

print("Modified_trX")

teX = teX_s_2.reshape(teX_rotated.__len__(), 28 * 28)
trX = trX_s_2.reshape(trX_rotated.__len__(), 28 * 28)

model = KNNClassifier(4, "my_own", "cosine", True)

model.fit(trX, trY)
result = model.predict(teX)

print("Accuracy of best method is: ", ac_s(teY, result))

f = open("save_file_all.txt", "w")
for item in result:
    f.write(str(item))
