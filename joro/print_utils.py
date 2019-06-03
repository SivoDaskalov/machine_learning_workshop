def print_ndarray(input):
    n, p = input.shape
    for i in range(n):
        row_values = input[i, :]
        res1 = ['%.2f' % value if value >= 0 else "NEG" for value in row_values]  # list comprehension

        res2 = []
        for value in row_values:
            if value >= 0:
                transformed_value = '%.2f' % value
            else:
                transformed_value = 'NEG'
            res2.append(transformed_value)

        res3 = []
        for j in range(len(row_values)):
            value = row_values[j]
            if value >= 0:
                transformed_value = '%.2f' % value
            else:
                transformed_value = 'NEG'
            res3.append(transformed_value)

        line = '\t'.join(res1)
        print(line)
    print('')
    print(input)
