// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -arch=sm_30 -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4


// Compile:
// g++ -c readppm.c milli.c
// nvcc -c filter.cu
// g++ filter.o readppm.o milli.o -lGL -lglut -L/usr/local/cuda/lib64 -lcudart
// ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 20
#define maxKernelSizeY 20

#define SUB_SIZE 16
#define FILTER_RAD_X 10
#define FILTER_RAD_Y 10


__device__ const unsigned int CACHE_SIZE_X = (2 * maxKernelSizeX + 1) * 3; // RGB
__device__ const unsigned int CACHE_SIZE_Y = 2 * maxKernelSizeY + 1;

__shared__ int cacheShared[CACHE_SIZE_X][CACHE_SIZE_Y];

__device__ void addPadding(unsigned char *image, int kernelsizex, int kernelsizey, int imagesizex, int imagesizey){

  int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

  for (int yy = 0; yy < blockDim.y + 2*kernelsizey; yy++) {
    for (int xx = 0; xx < (blockDim.x + 2 * kernelsizex); xx++) {

      int globalX = min(max(x + xx, 0), imagesizex - 1);
      int globalY = min(max(y + yy, 0), imagesizey - 1);

      cacheShared[xx * 3 + 0][yy] = image[( globalY*imagesizex + globalX ) * 3 + 0];
      cacheShared[xx * 3 + 1][yy] = image[( globalY*imagesizex + globalX ) * 3 + 1];
      cacheShared[xx * 3 + 2][yy] = image[( globalY*imagesizex + globalX ) * 3 + 2];
    }
  }
}

__global__ void naive_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

  int dy, dx;
  unsigned int sumx, sumy, sumz;

  int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!

	if (x < imagesizex && y < imagesizey) // If inside image
	{
// Filter kernel (simple box filter)
	sumx=0;sumy=0;sumz=0;
	for(dy=-kernelsizey;dy<=kernelsizey;dy++)
		for(dx=-kernelsizex;dx<=kernelsizex;dx++)
		{
			// Use max and min to avoid branching!
			int yy = min(max(y+dy, 0), imagesizey-1);
			int xx = min(max(x+dx, 0), imagesizex-1);

			sumx += image[((yy)*imagesizex+(xx))*3+0];
			sumy += image[((yy)*imagesizex+(xx))*3+1];
			sumz += image[((yy)*imagesizex+(xx))*3+2];
		}
	out[(y*imagesizex+x)*3+0] = sumx/divby;
	out[(y*imagesizex+x)*3+1] = sumy/divby;
	out[(y*imagesizex+x)*3+2] = sumz/divby;
	}
}

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex, const int kernelsizex, const int kernelsizey)
{
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	if (x > imagesizex) return;
	if (y > imagesizey) return;


  // Single thread padding
  if(threadIdx.x == 0 && threadIdx.y == 0){
    addPadding(image, kernelsizex, kernelsizey, imagesizex, imagesizey);
  }

  __syncthreads();

	// Print shared memory
  /*
	if (blockIdx.y == 0 && blockIdx.x == 7 && threadIdx.x == 0 && threadIdx.y == 0) {
		printf("\n");

		for (int yy = 0; yy < blockDim.y + 2*kernelsizey; yy++) {
			for (int xx = 0; xx < 3*(blockDim.x + 2 * kernelsizex); xx+=3) {

				if (cacheShared[xx][yy])
					printf("%d ", cacheShared[xx][yy]);//printf("# ");
				else
					printf(".  ");
			}
			printf("\n");
		}

		printf("\n\n");
	}
  */


	unsigned int divby = (2 * kernelsizex + 1)*(2 * kernelsizey + 1);

	unsigned int sumx, sumy, sumz;
	int localY = threadIdx.y + kernelsizey;
	int localX = threadIdx.x + kernelsizex;
	sumx = 0; sumy = 0; sumz = 0;


  // Print averaging for pixel 0
  /*
  if (blockIdx.y == 0 && blockIdx.x == 7 && threadIdx.x == 0 && threadIdx.y == 0) {
    for (int dy = -kernelsizey; dy <= kernelsizey; dy++) {
      for (int dx = -kernelsizex; dx <= kernelsizex; dx++) {

        int yy = localY + dy;
        int xx = localX + dx;

        printf("%d ", cacheShared[xx * 3 + 0][yy]);

      }
      printf("\n");
    }
  }
  */


	//if (localX < blockDim.x + kernelsizex && localY < blockDim.y + kernelsizey &&
	//	localX > kernelsizex && localY > kernelsizey) { // If inside kernel
		for (int dy = -kernelsizey; dy <= kernelsizey; dy++) {
			for (int dx = -kernelsizex; dx <= kernelsizex; dx++) {

				int yy = localY + dy;
				int xx = localX + dx;

				sumx += cacheShared[xx * 3 + 0][yy];
				sumy += cacheShared[xx * 3 + 1][yy];
				sumz += cacheShared[xx * 3 + 2][yy];
			}
		}
	//}


	out[(y*imagesizex + x) * 3 + 0] = sumx / divby;
	out[(y*imagesizex + x) * 3 + 1] = sumy / divby;
	out[(y*imagesizex + x) * 3 + 2] = sumz / divby;

}

__global__ void gaussian_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex, const int kernelsizex, const int kernelsizey)
{
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	if (x > imagesizex) return;
	if (y > imagesizey) return;


  // Single thread padding
  if(threadIdx.x == 0 && threadIdx.y == 0){
    addPadding(image, kernelsizex, kernelsizey, imagesizex, imagesizey);
  }

  __syncthreads();


	unsigned int sumx, sumy, sumz;
	int localY = threadIdx.y + kernelsizey;
	int localX = threadIdx.x + kernelsizex;
	sumx = 0; sumy = 0; sumz = 0;

  const int gaussWeight[] = {1, 4, 6, 4, 1};
  unsigned int divby = 16;

	for (int dy = -kernelsizey; dy <= kernelsizey; dy++) {
		for (int dx = -kernelsizex; dx <= kernelsizex; dx++) {

			int yy = localY + dy;
			int xx = localX + dx;

      int index = (dx + kernelsizex) + (dy + kernelsizey);

			sumx += cacheShared[xx * 3 + 0][yy] * gaussWeight[index];
			sumy += cacheShared[xx * 3 + 1][yy] * gaussWeight[index];
			sumz += cacheShared[xx * 3 + 2][yy] * gaussWeight[index];
		}
	}

	out[(y*imagesizex + x) * 3 + 0] = sumx / divby;
	out[(y*imagesizex + x) * 3 + 1] = sumy / divby;
	out[(y*imagesizex + x) * 3 + 2] = sumz / divby;

}


__device__ int find_median(int arr[], int N){

  int i = 1;
  int j, tmp;
  while( i < N ){
    j=i;
    while(j>0 && arr[j-1] > arr[j]){
        tmp = arr[j];
        arr[j] = arr[j-1];
        arr[j-1] = tmp;

        j--;
    }
    i++;
  }

  return arr[N/2];
}

__global__ void median_filter(unsigned char *image, unsigned char *out, const unsigned int imagesizey, const unsigned int imagesizex, const int kernelsizex, const int kernelsizey)
{
  // map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	if (x > imagesizex) return;
	if (y > imagesizey) return;


  // Single thread padding
  if(threadIdx.x == 0 && threadIdx.y == 0){
    addPadding(image, kernelsizex, kernelsizey, imagesizex, imagesizey);
  }

  __syncthreads();


	unsigned int sumx, sumy, sumz;
	int localY = threadIdx.y + kernelsizey;
	int localX = threadIdx.x + kernelsizex;
	sumx = 0; sumy = 0; sumz = 0;

  int kernel_arrR[(maxKernelSizeX*2+1) + (maxKernelSizeY*2+1)];
  int kernel_arrG[(maxKernelSizeX*2+1) + (maxKernelSizeY*2+1)];
  int kernel_arrB[(maxKernelSizeX*2+1) + (maxKernelSizeY*2+1)];

	for (int dy = -kernelsizey; dy <= kernelsizey; dy++) {
		for (int dx = -kernelsizex; dx <= kernelsizex; dx++) {

			int yy = localY + dy;
			int xx = localX + dx;

      int index = (dx + kernelsizex) + (dy + kernelsizey);

			kernel_arrR[index] = cacheShared[xx * 3 + 0][yy];
			kernel_arrG[index] = cacheShared[xx * 3 + 1][yy];
			kernel_arrB[index] = cacheShared[xx * 3 + 2][yy];
		}
	}

  int medR = find_median(kernel_arrR, kernelsizex + kernelsizey);
  int medG = find_median(kernel_arrG, kernelsizex + kernelsizey);
  int medB = find_median(kernel_arrB, kernelsizex + kernelsizey);


	out[(y*imagesizex + x) * 3 + 0] = medR;
	out[(y*imagesizex + x) * 3 + 1] = medG;
	out[(y*imagesizex + x) * 3 + 2] = medB;

}

// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_temp_output, *dev_input;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}

	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
  cudaMalloc( (void**)&dev_temp_output, imagesizex*imagesizey*3);

  dim3 grid(imagesizex/SUB_SIZE,imagesizey/SUB_SIZE);
  dim3 block(SUB_SIZE, SUB_SIZE);
  printf("\n");

  // Naive version

  ResetMilli();
	naive_filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizey, imagesizex, kernelsizex, kernelsizey); // Awful load balance
  printf("Naive: \t\t\t%f\n", GetSeconds());
  cudaThreadSynchronize();


  /*
  // Task 1
  ResetMilli();
	filter<<<grid,block>>>(dev_input, dev_bitmap, imagesizey, imagesizex, kernelsizex, kernelsizey); // Awful load balance
  printf("Averaging: \t\t%f\n", GetSeconds());
  cudaThreadSynchronize();


  // Task 2 - seperable filter kernels
  ResetMilli();
  filter<<<grid,block>>>(dev_input, dev_temp_output, imagesizey, imagesizex, 0, kernelsizey);
  filter<<<grid,block>>>(dev_temp_output, dev_bitmap, imagesizey, imagesizex, kernelsizex, 0);
  printf("Avg. Seperable: \t%f\n", GetSeconds());
  cudaThreadSynchronize();

  // Task 3 - seperable gaussian
  ResetMilli();
  gaussian_filter<<<grid,block>>>(dev_input, dev_temp_output, imagesizey, imagesizex, 0, 2);
  gaussian_filter<<<grid,block>>>(dev_temp_output, dev_bitmap, imagesizey, imagesizex, 2, 0);
  printf("Gauss. Seperable: \t%f\n", GetSeconds());
  cudaThreadSynchronize();
  */
  // Task 4 - Median
  ResetMilli();
  median_filter<<<grid,block>>>(dev_input, dev_temp_output, imagesizey, imagesizex, 0, FILTER_RAD_Y);
  median_filter<<<grid,block>>>(dev_temp_output, dev_bitmap, imagesizey, imagesizex, FILTER_RAD_X, 0);
  printf("Medi. Seperable: \t%f\n", GetSeconds());
  cudaThreadSynchronize();




//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"maskros-noisy.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(FILTER_RAD_X, FILTER_RAD_Y);

// You can save the result to a file like this:
//	writeppm("out.ppm", imagesizey, imagesizex, pixels);

	glutMainLoop();
	return 0;
}
