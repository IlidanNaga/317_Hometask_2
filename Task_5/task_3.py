import numpy as np


class BatchGenerator:

    def __init__(self, list_of_sequences, batch_size, shuffle=False):

        sequences = []

        for item in list_of_sequences:
            if type(item) == list:
                sequences.append(item)
            else:
                sequences.append(item.tolist())

        self.shuffle = shuffle

        self.len = list_of_sequences[0].__len__()

        self.batch_am = int(self.len / batch_size)

        if self.batch_am != self.len / batch_size:
            self.batch_am += 1

        self.batched_sequences = []

        for val in range(self.batch_am):

            current_sequences = []

            for item in sequences:
                current_sequences.append(item[val * batch_size: (val + 1) * batch_size])

            self.batched_sequences.append(current_sequences)

    def __iter__(self):

        self.batch_num = 0

        self.indexes = np.arange(self.batch_am)

        if self.shuffle:
            np.random.shuffle(self.indexes)

        return self

    def __next__(self):

        if self.batch_num < self.batch_am:
            self.batch_num += 1
            return self.batched_sequences[self.indexes[self.batch_num - 1]]
        else:
            raise StopIteration
