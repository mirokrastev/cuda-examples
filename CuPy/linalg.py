import cupy as cp


array = cp.array([
    [1, 2, 3],
    [3, 2, -2],
    [4, -3, 5]
])

inverse_matrix = cp.linalg.inv(array)

identity_matrix = cp.matmul(array, inverse_matrix)

print(f'Determinant of matrix: {cp.linalg.det(array)}')
print(f'Inverse of matrix: {inverse_matrix}')
print(f'Identity matrix {identity_matrix}')
print(f'Rank of matrix: {cp.linalg.matrix_rank(array)}')

"""
<<< Determinant of matrix: -93
<<< Inverse of matrix: [...]
<<< Identity matrix: [1, 0, 0]
                     [0, 1, 0]
                     [0, 0, 1]
<<< Rank of matrix: 3
"""
