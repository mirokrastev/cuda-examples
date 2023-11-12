#include <iostream>
#define N 65536

using namespace std;

__global__ void vectorAddition(int* a, int* b, int* c) {
    int id =  blockDim.x * blockIdx.x + threadIdx.x;
    if(N >= id) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    const size_t vectorSize = N * sizeof(int);

    // Allocate memory for host vectors <> Vectors in main memory
    int* hostVectorA = (int*)malloc(vectorSize);
    int* hostVectorB = (int*)malloc(vectorSize);
    int* hostVectorC = (int*)malloc(vectorSize);

    // Allocate memory for device vectors <> Vectors in the GPU memory
    int *deviceVectorA, *deviceVectorB, *deviceVectorC;
    cudaMalloc(&deviceVectorA, vectorSize);
    cudaMalloc(&deviceVectorB, vectorSize);
    cudaMalloc(&deviceVectorC, vectorSize);

    for (int i = 0; i < N; i++) {
        hostVectorA[i] = i;
        hostVectorB[i] = i + 10;
    }

    // Copy data from CPU Array to GPU Array
    cudaMemcpy(deviceVectorA, hostVectorA, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorB, hostVectorB, vectorSize, cudaMemcpyHostToDevice);

    // Execution parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = N / threadsPerBlock;

    // Launch kernel
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceVectorB, deviceVectorC);

    // Copy data from GPU Array to CPU Array
    cudaMemcpy(hostVectorC, deviceVectorC, vectorSize, cudaMemcpyDeviceToHost);

    // Free CPU Memory
    free(hostVectorA);
    free(hostVectorB);
    free(hostVectorC);

    // Free GPU Memory
    cudaFree(deviceVectorA);
    cudaFree(deviceVectorB);
    cudaFree(deviceVectorC);
}
