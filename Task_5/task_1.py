import numpy as np


def encode_rle(vector):

    positions = np.flatnonzero(~np.isclose(vector[:-1], vector[1:])) + 1
    positions = np.append(0, positions)

    lengths = np.diff(np.append(positions, len(vector)))

    return vector[positions], lengths


class RleSequence:

    def __init__(self, input_sequence):
        self.encod_run, self.encod_len = encode_rle(input_sequence)

        self.len = len(input_sequence)
        self.encodings_len = len(self.encod_len)

    def showItems(self):
        print("Run: ", self.encod_run)
        print("Len: ", self.encod_len)
        print("Arr_len: ", self.len)

    def __getitem__(self, arg):

        if type(arg) == int:
            # print(arg)

            if arg < -1 * self.len or arg >= self.len:
                raise TypeError

            pos = arg if arg >= 0 else self.len + arg
            # print(pos)

            for i in range(self.encodings_len):
                pos -= self.encod_len[i]
                if pos < 0:
                    pos = i
                    break

            return self.encod_run[i]


        elif type(arg) == slice:

            if arg.start is None:
                cur_start = 0
            elif arg.start > 0:
                cur_start = arg.start
            else:
                cur_start = max(0, self.len + arg.start)

            if arg.stop is None:
                cur_stop = self.len
            elif arg.stop >= 0:
                cur_stop = min(self.len, arg.stop)
            else:
                cur_stop = self.len + arg.stop

            if arg.step is None:
                cur_step = 1
            else:
                cur_step = arg.step

            first_pos = 0
            first_val = self.encod_len[0]
            for i in range(1, self.encodings_len):

                if first_val >= cur_start:
                    first_pos = i - 1
                    break

                first_val += self.encod_len[i]

            # print("First Positions", first_val, first_pos)
            res_list = []

            for i in range(cur_start, cur_stop, cur_step):

                while i >= first_val:
                    first_pos += 1
                    first_val += self.encod_len[first_pos]

                # print(i, "res_ind ", first_pos, "res_cumsum ", first_val, self.encod_run[first_pos])
                res_list.append(self.encod_run[first_pos])

            return np.array(res_list)

    def __iter__(self):
        self.position = 0
        self.ans_position = 0
        return self

    def __next__(self):

        if self.position >= self.encod_len[self.ans_position]:
            self.position -= self.encod_len[self.ans_position]
            self.ans_position += 1

        self.position += 1

        if self.ans_position >= self.encodings_len:
            raise StopIteration

        return self.encod_run[self.ans_position]

    def __contains__(self, arg):
        if arg in self.encod_run:
            return True

        return False