
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// nvcc matrix_gpu.cu -o matrix_gpu && ./matrix_gpu

const int N = 1024;					// Size of matrix
const int gridsize = 16;	// Num blocks
const int blocksize = 64;	// Num threads per block


__global__ void add_matrix(float *a, float *b, float *c) {

	// Good coaslecing
	//int row = blockIdx.y * blockDim.y + threadIdx.y;
	//int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Bad
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N)
		c[row*N + col] = a[row*N + col] + b[row*N + col];

}

int main()
{
	float gpuTime = 0;

	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];

	float *d_a, *d_b, *d_c;

	// Allocate memory on device
	cudaMalloc((void**)&d_a, N * N * sizeof(float));
	cudaMalloc((void**)&d_b, N * N * sizeof(float));
	cudaMalloc((void**)&d_c, N * N * sizeof(float));

	// Initialize matrices
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			int index = i*N + j;
			a[index] = index;
			b[index] = 2*index;
		}
	}

	cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, N * N * sizeof(float), cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(gridsize, gridsize);
	dim3 numBlocks(blocksize, blocksize);

	// Timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Execute the kernel
	cudaEventRecord(start);
	add_matrix <<<numBlocks, threadsPerBlock >>> (d_a, d_b, d_c);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);


	// sync threads

	cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


	// Asert result is correct

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = i*N + j;
			if (c[index] != a[index] + b[index]) {
				printf("Error: %f + %f != %f\n", a[index], b[index], c[index]);
				return EXIT_FAILURE;
			}
		}
	}





	/*
	for (int i = 0; i < N; i++) {
		// A
		printf("|");
		for (int j = 0; j < N; j++) {
			printf("%.2f ", a[i + j*N]);
		}
		printf("|\t");

		// B
		printf("|");
		for (int j = 0; j < N; j++) {
			printf("%.2f ", b[i + j*N]);
		}
		printf("|\t");

		// C
		printf("|");
		for (int j = 0; j < N; j++) {
			printf("%.2f ", c[i + j*N]);
		}
		printf("|\t\n");

	}
	*/




	printf("Problem size: %i\t\n", N);
	printf("GPU Time: \t%f\n", gpuTime);


	delete[] a;
	delete[] b;
	delete[] c;

  return 0;
}
