def get_new_dictionary(input_dict_name, output_dict_name):

    file = open(input_dict_name, 'r')

    lines = file.readlines()
    lines.pop(0)
    formed_lines = []
    for string in lines[:-1]:
        formed_lines.append(string[:-1])

    formed_lines.append(lines[-1])

    draconic = []
    human = []
    for string in formed_lines:
        current = string.split(' - ')
        translation = current[1].split(', ')
        for item in translation:
            if item in draconic:
                human[draconic.index(item)].append(current[0])
            else:
                draconic.append(item)
                human.append([current[0]])

    for item in human:
        item.sort()

    final = []
    for i in range(draconic.__len__()):
        h_line = human[i][0]
        for item in human[i][1:]:
            h_line += ', ' + item
        line = draconic[i] + ' - ' + h_line
        final.append(line)

    final.sort()
    final.insert(0, str(final.__len__()))

    out_file = open(output_dict_name, 'w')
    for item in final:
        out_file.write(item+'\n')
    file.close()
    out_file.close()
    return
