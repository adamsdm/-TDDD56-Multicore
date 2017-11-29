
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16;
const int blocksize = 16;

__global__
void simple(float *a, float *b)
{
	b[threadIdx.x] = a[threadIdx.x] * a[threadIdx.x];
}

int main()
{
	float *a = new float[N];
	float *b = new float[N];
	float *da;
	float *db;
	const int size = N * sizeof(float);

	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&db, size);

	// Initialize data
	for (int i = 0; i < N; i++) {
		a[i] = i;
	}
	
	cudaMemcpy(da, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(blocksize, 1);
	dim3 dimGrid(1, 1);
	simple <<<dimGrid, dimBlock >>> (da, db);

	cudaMemcpy(b, db, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		printf("%.2f^2 = %.2f\n", a[i], b[i]);
	}

	
	return EXIT_SUCCESS;
}
