#include <iostream>
#include <cuda.h>

#define N 100000  // Size of the vectors

// CUDA kernel for SAXPY
__global__ void saxpy(float a, float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
	    y[idx] = a * x[idx] + y[idx];
	}
}

int main() {
    float a = 2.0f;  // Scalar multiplier
    float h_x[N], h_y[N];
	float *d_x, *d_y;

    // Initialize vectors on host
    for (int i = 0; i < N; i++) {
        h_x[i] = i * 0.5f;
        h_y[i] = i * 0.1f;
    }

    // Allocate memory on device
    cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the SAXPY kernel
    int blockSize = 256;
	int gridSize = (N + blockSize - 1) / blockSize;
	saxpy<<<gridSize, blockSize>>>(a, d_x, d_y, N);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print part of the result
    std::cout << "First 5 results of SAXPY operation:" << std::endl;
	for (int i = 0; i < 5; i++) {
	    std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
	}

    // Free device memory
    cudaFree(d_x);
	cudaFree(d_y);

    return 0;
}
    