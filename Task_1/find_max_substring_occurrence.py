def find_max_substring_occurrence(input_str):

    dividers = []
    for i in range(1, input_str.__len__() + 1):
        if input_str.__len__() % i == 0:
            dividers.append(i)

    substr_list = []
    for i in dividers:
        for j in range(input_str.__len__() - i + 1):
            k = j + i
            substr_list.append(input_str[j: k])

    max_count = 0

    for substr in substr_list:

        num = input_str.count(substr)

        if num * substr.__len__() == input_str.__len__() and num > max_count:
            max_count = num

    return max_count
