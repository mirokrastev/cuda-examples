from numba import cuda
import numpy as np


@cuda.jit
def matrix_addition(a, b, c):
    row, col = cuda.grid(2)
    if row < c.shape[0] and col < c.shape[1]:
        c[row, col] = a[row, col] + b[row, col]


def main():
    # Matrix dimensions
    num_rows = 50
    num_cols = 50

    # Create random matrices A and B
    host_mat_a = np.random.rand(num_rows, num_cols).astype(np.float32)
    host_mat_b = np.random.rand(num_rows, num_cols).astype(np.float32)
    host_mat_c = np.zeros((num_rows, num_cols), dtype=np.float32)

    # Copy the matrices to the GPU
    device_mat_a = cuda.to_device(host_mat_a)
    device_mat_b = cuda.to_device(host_mat_b)
    device_mat_c = cuda.device_array_like(host_mat_c)

    # Define the grid and block dimensions for the kernel
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (num_rows + threads_per_block[0] - 1) // threads_per_block[0],
        (num_cols + threads_per_block[1] - 1) // threads_per_block[1],
    )

    # Launch the matrix addition kernel on the GPU
    matrix_addition[blocks_per_grid, threads_per_block](device_mat_a, device_mat_b, device_mat_c)

    # Copy the result back from the GPU
    device_mat_c.copy_to_host(host_mat_c)


if __name__ == "__main__":
    main()
