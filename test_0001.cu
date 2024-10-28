#include <stdio.h>

__global__ void cuda_hello() {
	    printf("Hello from the GPU!\n");
}

int main() {
    // Launch a CUDA kernel with one block and one thread
    cuda_hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}




