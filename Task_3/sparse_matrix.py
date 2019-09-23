class CooSparseMatrix:

    def __init__(self, ijx_list, shape):

        self.value_dict = {}
        self.x_max = shape[0]   # amount of columns
        self.y_max = shape[1]   # amount of lines

        self.shape_flag = True
        self.size = shape[0] * shape[1]

        self.shape = shape

        for item in ijx_list:

            i, j, x = item[0], item[1], item[2]
            pos = self.compute_pos(i, j)

            if pos in self.value_dict:

                raise TypeError

            if x != 0:
                self.value_dict[pos] = x

    def compute_pos(self, i, j):

        if i > self.x_max or j > self.y_max:
            raise TypeError

        return self.x_max * i + j

    def return_limits(self, line_num):

        if line_num >= self.y_max:
            raise TypeError

        return line_num * self.x_max, (line_num + 1) * self.x_max - 1

    def return_coords(self, pos):

        return pos // self.y_max, pos % self.x_max

    def return_pos(self, line_num, column_num):

        if line_num > self.x_max or column_num > self.y_max:
            raise TypeError

        return line_num * self.y_max + column_num

    def status(self):

        print('Dict of all coordinates values', self.value_dict)
        print('Shape: ', self.x_max, self.y_max)

        for i in range(self.x_max):
            line = []
            for j in range(self.y_max):
                line.append(self[i, j])
            print(len(line), line)

    def __getitem__(self, args):

        if type(args) is int:
            length = 1
        elif type(args) is tuple:
            length = len(args)
        else:
            raise TypeError

        if length == 1:

            min_coord, max_coord = self.return_limits(args)
            line = []
            for item in range(min_coord, max_coord + 1):

                x, y = self.return_coords(item)

                if item in self.value_dict:
                    line.append((0, y, self.value_dict.get(item)))

            return CooSparseMatrix(line, shape=(1, self.x_max))

        elif length == 2:

            pos = self.return_pos(args[0], args[1])
            if pos in self.value_dict:
                return self.value_dict.get(pos)
            return 0

        else:
            raise TypeError

    def __setitem__(self, key, value):

        if key[0] > self.y_max or key[1] > self.x_max:
            raise TypeError

        pos = key[0] * self.y_max + key[1]

        if pos in self.value_dict:
            if value == 0:
                self.value_dict.pop(pos)
            else:
                self.value_dict[pos] = value

        else:

            if value != 0:
                self.value_dict[pos] = value

    def __add__(self, other):

        if type(other) != CooSparseMatrix:
            raise TypeError

        if other.x_max != self.x_max or other.y_max != self.y_max:
            raise TypeError

        # В общем, тут надо собрать новую матрицу, а не модифицировать старую, тогда всё будет ок
        new_dict = self.value_dict.copy()

        for key in other.value_dict:

            if key in new_dict:
                new_dict[key] += other.value_dict[key]
            else:
                new_dict[key] = other.value_dict[key]

        line = []
        for key in new_dict:

            x, y = self.return_coords(key)

            line.append((x, y, new_dict.get(key)))

        return CooSparseMatrix(line, shape=(self.y_max, self.x_max))

    def __sub__(self, other):

        if type(other) != CooSparseMatrix:
            raise TypeError

        if other.x_max != self.x_max or other.y_max != self.y_max:
            raise TypeError

        new_dict = self.value_dict.copy()

        for key in other.value_dict:

            if key in new_dict:
                new_dict[key] -= other.value_dict[key]
            else:
                new_dict[key] = -1 * other.value_dict[key]

        line = []
        for key in new_dict:
            x, y = self.return_coords(key)

            line.append((x, y, new_dict.get(key)))

        return CooSparseMatrix(line, shape=(self.y_max, self.x_max))

    def __mul__(self, other):

        if type(other) != int and type(other) != float:
            raise TypeError

        new_dict = self.value_dict.copy()

        if other == 0:
            new_dict = {}
        else:
            for key in new_dict:
                new_dict[key] *= other

        line = []
        for key in new_dict:
            x, y = self.return_coords(key)

            line.append((x, y, new_dict.get(key)))

        return CooSparseMatrix(line, shape=(self.y_max, self.x_max))

    def __rmul__(self, other):

        return self.__mul__(other)

    def __setattr__(self, key, value):

        if key != 'shape' and key != 'T':
            super().__setattr__(key, value)
        elif key == 'shape':

            if self.shape_flag:
                self.shape_flag = False
                super().__setattr__(key, value)

            else:
                if type(value) != tuple:
                    raise TypeError

                if len(value) != 2:
                    raise TypeError

                if type(value[0]) != int or type(value[1]) != int:
                    raise TypeError

                if value[0] * value[1] != self.size:
                    raise TypeError

                super().__setattr__(key, value)
                self.x_max = value[0]
                self.y_max = value[1]
        else:
            raise AttributeError

    def __getattr__(self, item):

        if item != 'T':
            print(item)
            super().__getattribute__(item)
        else:

            new_shape = (self.shape[1], self.shape[0])
            igx_line = []

            for key in self.value_dict:

                x, y = self.return_coords(key)
                igx_line.append((y, x, self.value_dict.get(key)))

            return CooSparseMatrix(igx_line, new_shape)
