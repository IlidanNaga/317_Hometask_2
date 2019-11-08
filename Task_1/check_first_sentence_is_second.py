def check_first_sentence_is_second(first_str, second_str):
    first_list = first_str.split(' ')
    second_list = second_str.split(' ')

    for word in second_list:

        if word not in first_list:
            return False
        else:
            first_list.remove(word)

    return True
