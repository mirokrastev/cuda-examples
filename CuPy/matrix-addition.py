import cupy as cp


def main():
    # Matrix dimensions
    num_rows = 50
    num_cols = 50

    # Create random matrices A and B ON device
    device_matrix_a = cp.random.rand(num_rows, num_cols).astype(cp.float32)
    device_matrix_b = cp.random.rand(num_rows, num_cols).astype(cp.float32)

    # Do matrix addition with the built-in function
    device_result = cp.add(device_matrix_a, device_matrix_b)

    # Copy result to CPU
    host_result = cp.asnumpy(device_result)


if __name__ == "__main__":
    main()
