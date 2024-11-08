#include <iostream>
#include <cuda.h>

#define N 512  // Size of the array

// CUDA kernel for reduction (sum)
__global__ void reduceSum(int *input, int *output, int size) {
    __shared__ int sharedData[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < size)
	    sharedData[tid] = input[idx];
	else
		sharedData[tid] = 0;
	__syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
	    if (tid < stride) {
		    sharedData[tid] += sharedData[tid + stride];
		}
	__syncthreads();
	}

    // Write the result for this block to global memory
    if (tid == 0)
	    output[blockIdx.x] = sharedData[0];
}

int main() {
    int h_input[N], h_output[256];
    int *d_input, *d_output;

    // Initialize input array with values
    for (int i = 0; i < N; i++) {
	    h_input[i] = i + 1;  // Array contains {1, 2, 3, ..., N}
	}

    // Device memory allocation
    cudaMalloc(&d_input, N * sizeof(int));
	cudaMalloc(&d_output, 256 * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch reduction kernel
    int blockSize = 256;
	int gridSize = (N + blockSize - 1) / blockSize;
	reduceSum<<<gridSize, blockSize>>>(d_input, d_output, N);

    // Copy partial results back to host
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Perform final reduction on CPU
    int sum = 0;
	for (int i = 0; i < gridSize; i++) {
	    sum += h_output[i];
	}

    // Print the sum
    std::cout << "Sum of array elements = " << sum << std::endl;

    // Free device memory
    cudaFree(d_input);
	cudaFree(d_output);

    return 0;
    }
    