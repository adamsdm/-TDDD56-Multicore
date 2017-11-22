
class CLWrapperClass_median_precompiled_Overlap2DKernel_median_kernel
{
public:
	
	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}
	
	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;
		
		std::string source = skepu2::backend::cl_helpers::replaceSizeT(R"###(
#define SKEPU_USING_BACKEND_CL 1

typedef struct{
	size_t i;
} index1_t;

typedef struct {
	size_t row;
	size_t col;
} index2_t;

size_t get_device_id()
{
	return SKEPU_INTERNAL_DEVICE_ID;
}

#define VARIANT_OPENCL(block) block
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block)

static unsigned char median_kernel(int ox, int oy, size_t stride, __local const unsigned char * image, size_t elemPerPx)
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


__kernel void median_precompiled_Overlap2DKernel_median_kernel(
	__global unsigned char* input, size_t elemPerPx,  __global unsigned char* output,
	size_t out_rows, size_t out_cols, size_t overlap_y, size_t overlap_x,
	size_t in_pitch, size_t sharedRows, size_t sharedCols,
	__local unsigned char* sdata)
{
	size_t xx = ((size_t)(get_global_id(0) / get_local_size(0))) * get_local_size(0);
	size_t yy = ((size_t)(get_global_id(1) / get_local_size(1))) * get_local_size(1);
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	

	if (x < out_cols + overlap_x * 2 && y < out_rows + overlap_y * 2)
	{
		size_t sharedIdx = get_local_id(1) * sharedCols + get_local_id(0);
		sdata[sharedIdx]= input[y * in_pitch + x];
		
		size_t shared_x = get_local_id(0)+get_local_size(0);
		size_t shared_y = get_local_id(1);
		while (shared_y < sharedRows)
		{
			while (shared_x < sharedCols)
			{
				sharedIdx = shared_y * sharedCols + shared_x; 
				sdata[sharedIdx] = input[(yy + shared_y) * in_pitch + xx + shared_x];
				shared_x = shared_x + get_local_size(0);
			}
			shared_x = get_local_id(0);
			shared_y = shared_y + get_local_size(1);
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (x < out_cols && y < out_rows)
		output[y * out_cols + x] = median_kernel(overlap_x, overlap_y,
			sharedCols, &sdata[(get_local_id(1) + overlap_y) * sharedCols + (get_local_id(0) + overlap_x)] , elemPerPx);
}
)###");
		
		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu2::backend::Device_CL *device : skepu2::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu2::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "median_precompiled_Overlap2DKernel_median_kernel", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 2D kernel '" << "median_precompiled_Overlap2DKernel_median_kernel" << "'");
			
			kernels(counter++, &kernel);
		}
		
		initialized = true;
	}
	
	static void mapOverlap2D
	(
		size_t deviceID, size_t localSize[2], size_t globalSize[2],
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *input, size_t elemPerPx, 
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *output,
		size_t out_rows, size_t out_cols, size_t overlap_y, size_t overlap_x,
		size_t in_pitch, size_t sharedRows, size_t sharedCols,
		size_t sharedMemSize
	)
	{
		skepu2::backend::cl_helpers::setKernelArgs(kernels(deviceID), input->getDeviceDataPointer(), elemPerPx,  output->getDeviceDataPointer(),
			out_rows, out_cols, overlap_y, overlap_x, in_pitch, sharedRows, sharedCols);
		clSetKernelArg(kernels(deviceID), 1 + 9, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernels(deviceID), 2, NULL, globalSize, localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 2D kernel");
	}
};
