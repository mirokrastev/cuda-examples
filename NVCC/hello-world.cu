# include <iostream>

using namespace std;

__global__ void helloFromGPU() {
    printf("Hello from the GPU!\n");
}

int main() {
    cout << "Hello from the CPU!" << endl;
    helloFromGPU<<<1, 1>>>();
}

/*
<<< Hello from the CPU!
<<< Hello from the GPU!
*/
