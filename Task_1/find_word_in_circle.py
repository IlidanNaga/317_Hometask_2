def find_word_in_circle(big_str, target_str):

    def check_symbol(case, target):

        if case == target:
            return True

        return False

    for iterable, symbol in enumerate(big_str):

        if symbol == target_str[0]:
            target_counter = 1
            circle_iterable = iterable

            while target_counter != target_str.__len__():

                circle_iterable = (circle_iterable + 1) % big_str.__len__()

                if not check_symbol(big_str[circle_iterable],
                                    target_str[target_counter]):
                    break
                else:
                    target_counter += 1

            if target_counter == target_str.__len__():
                return iterable, 1

            target_counter = 1
            circle_iterable = iterable

            while target_counter != target_str.__len__():

                circle_iterable = (circle_iterable - 1) % big_str.__len__()

                if not check_symbol(big_str[circle_iterable],
                                    target_str[target_counter]):
                    break
                else:
                    target_counter += 1

            if target_counter == target_str.__len__():
                return iterable, -1

    return -1
