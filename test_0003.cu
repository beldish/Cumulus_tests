#include <iostream>
#include <cuda.h>

#define N 16  // Size of the matrix

// CUDA kernel for matrix multiplication
__global__ void matrixMul(int *A, int *B, int *C, int width) {
	   int row = blockIdx.y * blockDim.y + threadIdx.y;
	   int col = blockIdx.x * blockDim.x + threadIdx.x;

	   int sum = 0;
	   if (row < width && col < width) {
	      for (int i = 0; i < width; i++) {
	      	  sum += A[row * width + i] * B[i * width + col];
	      }
	      C[row * width + col] = sum;
	   }
}

int main() {
	    int size = N * N * sizeof(int);
	        
	    // Host matrices
	    int h_A[N][N], h_B[N][N], h_C[N][N];

	    // Initialize host matrices with some values
	    for (int i = 0; i < N; i++) {
	    	for (int j = 0; j < N; j++) {
		    h_A[i][j] = i + j;
		    h_B[i][j] = i - j;
		}
	    }

	    // Device memory
	    int *d_A, *d_B, *d_C;
	    cudaMalloc(&d_A, size);
	    cudaMalloc(&d_B, size);
	    cudaMalloc(&d_C, size);

	    // Copy host data to device
	    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	    // Define grid and block dimensions
	    dim3 threadsPerBlock(16, 16);
	    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
			       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	    // Launch matrix multiplication kernel
	    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

	    // Copy result back to host
	    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	    // Print the resulting matrix
	    std::cout << "Result matrix C:" << std::endl;
	    for (int i = 0; i < N; i++) {
	    	for (int j = 0; j < N; j++) {
		    std::cout << h_C[i][j] << " ";
		 }
		 std::cout << std::endl;
	    }

	    // Free device memory
	    cudaFree(d_A);
	    cudaFree(d_B);
	    cudaFree(d_C);

	    return 0;
}

