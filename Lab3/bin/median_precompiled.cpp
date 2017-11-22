#ifndef SKEPU_PRECOMPILED
#define SKEPU_PRECOMPILED
#endif
#ifndef SKEPU_OPENMP
#define SKEPU_OPENMP
#endif
#ifndef SKEPU_OPENCL
#define SKEPU_OPENCL
#endif
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu2.hpp>

#include "support.h"


unsigned char median_kernel(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	float median;

	size_t size = (ox*2+1)*(oy*2+1);

	// Convert to 1D-Array
	float values[size];

	int a = 0;
	for (int y = -oy; y <= oy; ++y){
		for (int x = -ox; x <= ox; x += elemPerPx){
				values[a] = image[y*(int)stride+x];
				a++;
		}
	}



	// Insertion sort
	int i=0, j=0, tmp;
	while(i<size){

		j=i;
		while(j>0 && values[j-1] > values[j]){
			//swap
			tmp = values[j];
			values[j] = values[j-1];
			values[j-1] = tmp;
			j=j-1;
		}
		i = i+1;
	}

	median = values[size/2];

	return median;

	/*
	float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx)
			res += image[y*(int)stride+x];
	return res * scaling;
	*/
}
struct skepu2_userfunction_calculateMedian_median_kernel
{
constexpr static size_t totalArity = 5;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<int, int, size_t, const unsigned char *>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<size_t>;
constexpr static skepu2::AccessMode anyAccessMode[] = {
};

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	float median;

	size_t size = (ox*2+1)*(oy*2+1);

	// Convert to 1D-Array
	float values[size];

	int a = 0;
	for (int y = -oy; y <= oy; ++y){
		for (int x = -ox; x <= ox; x += elemPerPx){
				values[a] = image[y*(int)stride+x];
				a++;
		}
	}



	// Insertion sort
	int i=0, j=0, tmp;
	while(i<size){

		j=i;
		while(j>0 && values[j-1] > values[j]){
			//swap
			tmp = values[j];
			values[j] = values[j-1];
			values[j-1] = tmp;
			j=j-1;
		}
		i = i+1;
	}

	median = values[size/2];

	return median;

	/*
	float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx)
			res += image[y*(int)stride+x];
	return res * scaling;
	*/
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	float median;

	size_t size = (ox*2+1)*(oy*2+1);

	// Convert to 1D-Array
	float values[size];

	int a = 0;
	for (int y = -oy; y <= oy; ++y){
		for (int x = -ox; x <= ox; x += elemPerPx){
				values[a] = image[y*(int)stride+x];
				a++;
		}
	}



	// Insertion sort
	int i=0, j=0, tmp;
	while(i<size){

		j=i;
		while(j>0 && values[j-1] > values[j]){
			//swap
			tmp = values[j];
			values[j] = values[j-1];
			values[j-1] = tmp;
			j=j-1;
		}
		i = i+1;
	}

	median = values[size/2];

	return median;

	/*
	float scaling = 1.0 / ((ox/elemPerPx*2+1)*(oy*2+1));
	float res = 0;
	for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx)
			res += image[y*(int)stride+x];
	return res * scaling;
	*/
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_calculateMedian_median_kernel::anyAccessMode[];





#include "median_precompiled_Overlap2DKernel_median_kernel_cl_source.inl"
int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;

	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << "input output radius [backend]\n";
		exit(1);
	}

	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[4])};

	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";

	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu2::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu2::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);

	// Skeleton instance
	skepu2::backend::MapOverlap2D<skepu2_userfunction_calculateMedian_median_kernel, bool, CLWrapperClass_median_precompiled_Overlap2DKernel_median_kernel> calculateMedian(false);
	calculateMedian.setBackend(spec);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);

	auto timeTaken = skepu2::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);

	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";

	return 0;
}
