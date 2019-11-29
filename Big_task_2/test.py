import oracles
import numpy as np
from scipy import sparse


np.random.seed(4181)
l2_coef = np.random.randint(0, 10)
l, d = 1000, 10
my_oracle = oracles.BinaryLogistic(l2_coef=l2_coef)
X = sparse.csr_matrix(np.random.random((l, d)))
y = np.random.randint(0, 2, l) * 2 - 1
w = np.random.random(d)
res = my_oracle.func(X, y, w)
print(res)
