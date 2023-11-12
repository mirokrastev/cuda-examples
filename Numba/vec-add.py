from numba import cuda
import numpy as np


@cuda.jit
def vector_addition(a, b, c):
    id = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    if N >= id:
        c[id] = a[id] + b[id]


def main():
    # Define numpy arrays. Similar to list(range(N))
    host_vector_a, host_vector_b = np.arange(N), np.arange(N, step=10)
    host_vector_c = np.empty_like(host_vector_a)

    # Allocate memory for device vectors and copy data from CPU to GPU
    device_vector_a = cuda.to_device(host_vector_a)
    device_vector_b = cuda.to_device(host_vector_b)
    device_vector_c = cuda.device_array_like(host_vector_c)

    # Execution parameters
    threads_per_block = 256
    blocks_per_grid = N // threads_per_block

    # Launch kernel
    vector_addition[blocks_per_grid, threads_per_block](device_vector_a, device_vector_b, device_vector_c)

    # Copy data from GPU Array to CPU Array
    device_vector_c.copy_to_host(host_vector_c)


N = 65536
main()
