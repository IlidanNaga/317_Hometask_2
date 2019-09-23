class Polynomial:

    def __init__(self, *args):

        self.storage = []

        for item in args:

            self.storage.append(item)

    def __call__(self, pos):

        s = 0

        for it, item in enumerate(self.storage):

            s += item * pos ** it

        return s



