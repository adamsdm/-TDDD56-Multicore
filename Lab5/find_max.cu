#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>



// Reduction lab, find maximum

#define GRIDDIM		8
#define BLOCKDIM	1024

#define SIZE 1024*8
#define INIT_RAND
int data[SIZE];
int data2[SIZE];

/*
#define SIZE 20
int data[SIZE] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12 , 13, 14, 15, 16, 17, 18, 19, 20 };
int data2[SIZE] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12 , 13, 14, 15, 16, 17, 18, 19, 20 };
*/

/*
#define SIZE 16
int data[SIZE] = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE] = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
*/

__global__ void find_max(int *data, int N)
{
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int m = data[0];

	int chunksize = N / blockDim.x / gridDim.x;

	// Find max in each chunk
	for (int i = 0; i < chunksize; i++) {
		if (data[chunksize * idx + i] > m) {
			m = data[chunksize * idx + i];
		}
	}


	data[chunksize*idx] = m;
	

	// Let thread 0 at each block do the reduction
	if (idx == 0) {
		for (int i = 0; i < (blockDim.x * gridDim.x); i++) {
			if (data[idx + i * chunksize] > m) {
				m = data[idx + i * chunksize];
			}
		}
		data[idx] = m;
	}
}

void launch_cuda_kernel(int *data, int N)
{
	// Handle your CUDA kernel launches in this function

	int *devdata;
	int size = sizeof(int) * N;
	cudaMalloc((void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice);

	// Dummy launch
	dim3 dimBlock(BLOCKDIM, 1);
	dim3 dimGrid(GRIDDIM, 1);
	find_max <<<dimGrid, dimBlock >>>(devdata, N);
	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(devdata);
}

// CPU max finder (sequential)
void find_max_cpu(int *data, int N)
{
	int i, m;

	m = data[0];
	for (i = 0; i<N; i++) // Loop over data
	{
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}


int main()
{
	// Generate 2 copies of random data
	//srand(time(NULL));
	
#ifdef INIT_RAND
	for (long i = 0; i<SIZE; i++)
	{
		data[i] = rand() % (SIZE * 5);
		data2[i] = data[i];
	}
#endif // INIT_RAND

	
	// The GPU will not easily beat the CPU here!
	// Reduction needs optimizing or it will be slow.
	//ResetMilli();
	find_max_cpu(data, SIZE);
	//printf("CPU time %f\n", GetSeconds());
	//ResetMilli();
	launch_cuda_kernel(data2, SIZE);
	//printf("GPU time %f\n", GetSeconds());


	// Print result
	printf("\n");
	printf("CPU found max %d\n", data[0]);
	printf("GPU found max %d\n", data2[0]);
}
