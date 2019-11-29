import numpy as np
from time import time
from scipy import sparse
import scipy
from random import shuffle
from random import seed

from oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function='binary_logistic', step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
                |
        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """

        if loss_function != 'binary_logistic':
            raise TypeError

        self.loss = loss_function

        self.step_alpha = step_alpha
        self.step_beta = step_beta

        self.tolerance = tolerance

        self.max_iter = max_iter

        self.oracle = BinaryLogistic(l2_coef=kwargs['l2_coef'])

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        def lambda_k(k):
            return self.step_alpha / (k ** self.step_beta)

        if w_0 is None:
            w = np.zeros(X.shape[1])
        else:
            w = w_0

        history = {}
        times = []
        funcs = []
        ws = []

        cur_func = self.oracle.func(X, y, w)
        funcs.append(cur_func)
        times.append(0)
        ws.append(w)

        i = 1
        while i <= self.max_iter:

            time_b4 = time()

            w_next = w - lambda_k(i) * self.oracle.grad(X, y, w)

            next_func = self.oracle.func(X, y, w_next)

            times.append(time() - time_b4)
            funcs.append(next_func)
            ws.append(w_next)

            if abs(next_func - cur_func) < self.tolerance:
                break

            w = w_next
            cur_func = next_func
            i += 1

        history['time'] = times
        history['func'] = funcs
        history['w'] = ws

        self.w = w_next

        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """

        if type(X) == scipy.sparse.csr.csr_matrix:
            # не знаю почему, но при посылке sparse matrix она ведет себя, как вектор, и np.dot возвращает число
            # это обрбатывает её отдельно
            ans = np.sign(X * self.w)
            ans[ans == 0] = -1
            return ans

        else:

            ans = np.sign(np.sum(X * self.w, axis=1))
            ans[ans == 0] = 1
            return ans

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """

        ans = X * self.w
        return 1 / (1 + np.exp(-1 * ans))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, batch_size = 1, loss_function = 'binary_logistic', step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        
        max_iter - максимальное число итераций (эпох)
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """

        if loss_function != "binary_logistic":
            raise TypeError

        self.oracle = BinaryLogistic(kwargs["l2_coef"])

        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.seed = random_seed
        
    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """

        seed(self.seed)
        np.random.seed(self.seed)

        def lambda_k(k):
            return self.step_alpha / ((k + 1) ** self.step_beta)

        if w_0 is None:
            w = np.zeros(X.shape[1])
        else:
            w = w_0

        # let's make it forbidden
        history = {}
        funcs = []
        times = []
        epoch = []
        weights = []

        cur_func = self.oracle.func(X, y, w)

        funcs.append(cur_func)
        times.append(0)
        epoch.append(0)
        weights.append(w)

        indexes = np.arange(X.shape[0])

        delta = self.batch_size / X.shape[0]
        threshold = X.shape[0] * log_freq

        i = 0
        amount = 0
        w_next = w

        while i < self.max_iter:

            if amount == 0:
                time_b4 = time()

            shuffle(indexes)
            currently_used = indexes[:self.batch_size]

            w_next = w - lambda_k(i) * self.oracle.grad(X[currently_used], y[currently_used], w)

            next_func = self.oracle.func(X, y, w_next)

            amount += self.batch_size

            if amount > threshold:
                funcs.append(next_func)
                times.append(time() - time_b4)
                epoch.append(i)
                weights.append(w_next)
                amount = 0

            if abs(next_func - cur_func) < self.tolerance:
                break

            i += delta
            w = w_next
            cur_func = next_func

        history['epoch_num'] = epoch
        history['time'] = times
        history['func'] = funcs
        history['weights'] = weights

        self.w = w_next

        if trace:
            return history