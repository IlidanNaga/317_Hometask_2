class WordContextGenerator:

    # при window_size = k окно берется k+1
    def __init__(self, words, window_size):

        self.result = []
        for pos in range(words.__len__()):
            # print(words[max(pos - window_size, 0) : pos])
            for item in words[max(pos - window_size, 0): pos]:
                # print(words[pos], item)
                self.result.append([words[pos], item])
                #self.result.append(words[pos] + ', ' + item)

            # print(words[pos + 1 : min(pos + window_size + 1, words.__len__())])
            for item in words[pos + 1: min(pos + window_size + 1, words.__len__())]:
                # print(words[pos], item)
                self.result.append([words[pos], item])
                # self.result.append(words[pos] + ', ' + item)

        self.len = self.result.__len__()

    def __iter__(self):

        self.pos = 0
        return self

    def __next__(self):

        if self.pos < self.len:
            self.pos += 1
            return self.result[self.pos - 1]
        else:
            raise StopIteration
