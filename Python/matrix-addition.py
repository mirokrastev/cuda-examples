from random import random


def matrix_addition(a, b):
    num_rows, num_cols = len(a), len(a[0])
    result_matrix = []

    for row in range(num_rows):
        result_matrix.append([])
        for col in range(num_cols):
            c = a[row][col] + b[row][col]
            result_matrix[-1].append(c)
    return result_matrix


def main(num_rows, num_cols):
    matrix_a = [[random() for _ in range(num_cols)] for _ in range(num_rows)]
    matrix_b = [[random() for _ in range(num_cols)] for _ in range(num_rows)]

    result = matrix_addition(matrix_a, matrix_b)


if __name__ == '__main__':
    main(100, 100)
