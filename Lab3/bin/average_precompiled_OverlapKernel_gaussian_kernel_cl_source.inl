
class CLWrapperClass_average_precompiled_OverlapKernel_gaussian_kernel
{
public:
	
	enum
	{
		KERNEL_VECTOR = 0,
		KERNEL_MATRIX_ROW,
		KERNEL_MATRIX_COL,
		KERNEL_MATRIX_COL_MULTI,
		KERNEL_COUNT
	};
	
	static cl_kernel kernels(size_t deviceID, size_t kerneltype, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8][KERNEL_COUNT]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID][kerneltype] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID][kerneltype];
	}
	
	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;
		
		std::string source = skepu2::backend::cl_helpers::replaceSizeT(R"###(
typedef struct {
	__global float *data;
	size_t size;
} skepu_vec_proxy_float;

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

static unsigned char gaussian_kernel(int o, size_t stride, __local const unsigned char * m, skepu_vec_proxy_float stencil, size_t elemPerPx)
{
	float scaling = 1.0 / (o/elemPerPx*2+1) ;
	float res = 0;

	for(int i = -o; i <= o; i += elemPerPx){
		res += m[i * (int)stride ] * stencil.data[(i+o)/elemPerPx];
	}
	return res;
}


__kernel void average_precompiled_OverlapKernel_gaussian_kernel_Vector(
	__global unsigned char* input, __global skepu_vec_proxy_float *skepu_container_stencil, size_t skepu_size_stencil, size_t elemPerPx,  __global unsigned char* output,
	__global unsigned char* wrap, size_t n, size_t overlap, size_t out_offset,
	size_t out_numelements, int poly, unsigned char pad, __local unsigned char* sdata
)
{
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	skepu_vec_proxy_float stencil = { .data = skepu_container_stencil, .size = skepu_size_stencil };

	
	if (poly == 0)
	{
		sdata[overlap + tid] = (i < n) ? input[i] : pad;
		if (tid < overlap)
			sdata[tid] = (get_group_id(0) == 0) ? pad : input[i - overlap];
		
		if (tid >= get_local_size(0) - overlap)
			sdata[tid + 2 * overlap] = (get_group_id(0) != get_num_groups(0) - 1 && i + overlap < n) ? input[i + overlap] : pad;
	}
	else if (poly == 1)
	{
		if (i < n)
			sdata[overlap + tid] = input[i];
		else if (i - n < overlap)
			sdata[overlap + tid] = wrap[overlap + i - n];
		else
			sdata[overlap + tid] = pad;
		
		if (tid < overlap)
			sdata[tid] = (get_group_id(0) == 0) ? wrap[tid] : input[i - overlap];
		
		if (tid >= get_local_size(0) - overlap)
			sdata[tid + 2 * overlap] = (get_group_id(0) != get_num_groups(0) - 1 && i + overlap < n) ? input[i + overlap] : wrap[overlap + i + overlap - n];
	}
	else if (poly == 2)
	{
		sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];
		if (tid < overlap)
			sdata[tid] = (get_group_id(0) == 0) ? input[0] : input[i-overlap];
		
		if (tid >= get_local_size(0) - overlap)
			sdata[tid + 2 * overlap] = (get_group_id(0) != get_num_groups(0) - 1 && i + overlap < n) ? input[i + overlap] : input[n - 1];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (i >= out_offset && i < out_offset + out_numelements)
		output[i - out_offset] = gaussian_kernel(overlap, 1, &sdata[tid + overlap] , stencil, elemPerPx);
}

__kernel void average_precompiled_OverlapKernel_gaussian_kernel_MatRowWise(
	__global unsigned char* input, __global skepu_vec_proxy_float *skepu_container_stencil, size_t skepu_size_stencil, size_t elemPerPx,  __global unsigned char* output,
	__global unsigned char* wrap, size_t n, size_t overlap, size_t out_offset, size_t out_numelements,
	int poly, unsigned char pad, size_t blocksPerRow, size_t rowWidth, __local unsigned char* sdata
)
{
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex= 2 * overlap * (int)(get_group_id(0) / blocksPerRow);
	size_t tmp  = (get_group_id(0) % blocksPerRow);
	size_t tmp2 = (get_group_id(0) / blocksPerRow);
	skepu_vec_proxy_float stencil = { .data = skepu_container_stencil, .size = skepu_size_stencil };

	
	if (poly == 0)
	{
		sdata[overlap+tid] = (i < n) ? input[i] : pad;
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? pad : input[i-overlap];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (i+overlap < n) && tmp!=(blocksPerRow-1)) ? input[i+overlap] : pad;
	}
	else if (poly == 1)
	{
		if (i < n)
			sdata[overlap+tid] = input[i];
		else if (i-n < overlap)
			sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
		else
			sdata[overlap+tid] = pad;
		
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[i-overlap];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && i+overlap < n && tmp!=(blocksPerRow-1))
				? input[i+overlap] : wrap[overlap+wrapIndex+(tid+overlap-get_local_size(0))];
	}
	else if (poly == 2)
	{
		sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];
		if(tid < overlap)
			sdata[tid] = (tmp==0) ? input[tmp2*rowWidth] : input[i-overlap];
		
		if(tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (i+overlap < n) && (tmp!=(blocksPerRow-1)))
				? input[i+overlap] : input[(tmp2+1)*rowWidth-1];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((i >= out_offset) && (i < out_offset+out_numelements))
		output[i-out_offset] = gaussian_kernel(overlap, 1, &sdata[tid+overlap] , stencil, elemPerPx);
}

__kernel void average_precompiled_OverlapKernel_gaussian_kernel_MatColWise(
	__global unsigned char* input, __global skepu_vec_proxy_float *skepu_container_stencil, size_t skepu_size_stencil, size_t elemPerPx,  __global unsigned char* output,
	__global unsigned char* wrap, size_t n, size_t overlap, size_t out_offset, size_t out_numelements,
	int poly, unsigned char pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth, __local unsigned char* sdata
	)
{
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex= 2 * overlap * (int)(get_group_id(0)/blocksPerCol);
	size_t tmp= (get_group_id(0) % blocksPerCol);
	size_t tmp2= (get_group_id(0) / blocksPerCol);
	size_t arrInd = (tid + tmp*get_local_size(0))*rowWidth + tmp2;
	skepu_vec_proxy_float stencil = { .data = skepu_container_stencil, .size = skepu_size_stencil };

	
	if (poly == 0)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd] : pad;
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : pad;
	}
	else if (poly == 1)
	{
		if (i < n)
			sdata[overlap+tid] = input[arrInd];
		else if (i-n < overlap)
			sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
		else
			sdata[overlap+tid] = pad;
		
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[(arrInd-(overlap*rowWidth))];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1)))
				? input[(arrInd+(overlap*rowWidth))] : wrap[overlap+wrapIndex+(tid+overlap-get_local_size(0))];
	}
	else if (poly == 2)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd] : input[n-1];
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1)))
				? input[(arrInd+(overlap*rowWidth))] : input[tmp2+(colWidth-1)*rowWidth];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((arrInd >= out_offset) && (arrInd < out_offset+out_numelements))
		output[arrInd-out_offset] = gaussian_kernel(overlap, 1, &sdata[tid+overlap] , stencil, elemPerPx);
}

__kernel void average_precompiled_OverlapKernel_gaussian_kernel_MatColWiseMulti(
	__global unsigned char* input, __global skepu_vec_proxy_float *skepu_container_stencil, size_t skepu_size_stencil, size_t elemPerPx,  __global unsigned char* output,
	__global unsigned char* wrap, size_t n, size_t overlap, size_t in_offset, size_t out_numelements,
	int poly, int deviceType, unsigned char pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
	__local unsigned char* sdata
)
{
	size_t tid = get_local_id(0);
	size_t i   = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex = 2 * overlap * (int)(get_group_id(0)/blocksPerCol);
	size_t tmp  = (get_group_id(0) % blocksPerCol);
	size_t tmp2 = (get_group_id(0) / blocksPerCol);
	size_t arrInd = (tid + tmp*get_local_size(0))*rowWidth + tmp2;
	skepu_vec_proxy_float stencil = { .data = skepu_container_stencil, .size = skepu_size_stencil };

	
	if (poly == 0)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : pad;
		if (deviceType == -1)
		{
			if (tid < overlap)
				sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
			 
			if(tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
		}
		else if (deviceType == 0) 
		{
			if(tid < overlap)
				sdata[tid] = input[arrInd];
			
			if(tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
		}
		else if (deviceType == 1)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1)))
					? input[(arrInd+in_offset+(overlap*rowWidth))] : pad;
		}
	}
	else if (poly == 1)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : ((i-n < overlap) ? wrap[(i-n)+ (overlap * tmp2)] : pad);
		if (deviceType == -1)
		{
			if (tid < overlap)
				sdata[tid] = (tmp==0) ? wrap[tid+(overlap * tmp2)] : input[(arrInd-(overlap*rowWidth))];
			
			if (tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
		}
		else if (deviceType == 0)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
		}
		else if (deviceType == 1)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1)))
					? input[(arrInd+in_offset+(overlap*rowWidth))] : wrap[(overlap * tmp2)+(tid+overlap-get_local_size(0))];
		}
	}
	else if (poly == 2)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd + in_offset] : input[n + in_offset - 1];
		if (deviceType == -1)
		{
			if (tid < overlap)
				sdata[tid] = (tmp == 0) ? input[tmp2] : input[arrInd - overlap * rowWidth];
			
			if (tid >= get_local_size(0) - overlap)
				sdata[tid+2*overlap] = input[arrInd + in_offset + overlap * rowWidth];
		}
		else if (deviceType == 0)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= get_local_size(0) - overlap)
				sdata[tid+2*overlap] = input[arrInd + in_offset + overlap * rowWidth];
		}
		else if (deviceType == 1)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= get_local_size(0) - overlap)
				sdata[tid + 2 * overlap] = (get_group_id(0) != get_num_groups(0) - 1 && (arrInd + overlap * rowWidth < n) && (tmp != blocksPerCol - 1))
					? input[arrInd + in_offset + overlap * rowWidth] : input[tmp2 + in_offset + (colWidth - 1) * rowWidth];
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (arrInd < out_numelements )
		output[arrInd] = gaussian_kernel(overlap, 1, &sdata[tid+overlap] , stencil, elemPerPx);
}
)###");
		
		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu2::backend::Device_CL *device : skepu2::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu2::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel_vector = clCreateKernel(program, "average_precompiled_OverlapKernel_gaussian_kernel_Vector", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D vector kernel '" << "average_precompiled_OverlapKernel_gaussian_kernel" << "'");
			
			cl_kernel kernel_matrix_row = clCreateKernel(program, "average_precompiled_OverlapKernel_gaussian_kernel_MatRowWise", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix row-wise kernel '" << "average_precompiled_OverlapKernel_gaussian_kernel" << "'");
			
			cl_kernel kernel_matrix_col = clCreateKernel(program, "average_precompiled_OverlapKernel_gaussian_kernel_MatColWise", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix col-wise kernel '" << "average_precompiled_OverlapKernel_gaussian_kernel" << "'");
			
			cl_kernel kernel_matrix_col_multi = clCreateKernel(program, "average_precompiled_OverlapKernel_gaussian_kernel_MatColWiseMulti", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix col-wise multi kernel '" << "average_precompiled_OverlapKernel_gaussian_kernel" << "'");
			
			kernels(counter, KERNEL_VECTOR,           &kernel_vector);
			kernels(counter, KERNEL_MATRIX_ROW,       &kernel_matrix_row);
			kernels(counter, KERNEL_MATRIX_COL,       &kernel_matrix_col);
			kernels(counter, KERNEL_MATRIX_COL_MULTI, &kernel_matrix_col_multi);
			counter++;
		}
		
		initialized = true;
	}
	
	static void mapOverlapVector
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *input, std::tuple<skepu2::Vector<float> *, skepu2::backend::DeviceMemPointer_CL<float> *> skepu_container_stencil, size_t elemPerPx, 
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *output, skepu2::backend::DeviceMemPointer_CL<unsigned char> *wrap,
		size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, unsigned char pad,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_VECTOR);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), std::get<1>(skepu_container_stencil)->getDeviceDataPointer(), std::get<0>(skepu_container_stencil)->size(), elemPerPx,  output->getDeviceDataPointer(),
			wrap->getDeviceDataPointer(), n, overlap, out_offset, out_numelements, poly, pad);
		clSetKernelArg(kernel, 3 + 9, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D vector kernel");
	}
	
	static void mapOverlapMatrixRowWise
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *input, std::tuple<skepu2::Vector<float> *, skepu2::backend::DeviceMemPointer_CL<float> *> skepu_container_stencil, size_t elemPerPx, 
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *output, skepu2::backend::DeviceMemPointer_CL<unsigned char> *wrap,
		size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, unsigned char pad, size_t blocksPerRow, size_t rowWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_ROW);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), std::get<1>(skepu_container_stencil)->getDeviceDataPointer(), std::get<0>(skepu_container_stencil)->size(), elemPerPx,  output->getDeviceDataPointer(),
			wrap->getDeviceDataPointer(), n, overlap, out_offset, out_numelements, poly, pad, blocksPerRow, rowWidth);
		clSetKernelArg(kernel, 3 + 11, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix row-wise kernel");
	}
	
	static void mapOverlapMatrixColWise
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *input, std::tuple<skepu2::Vector<float> *, skepu2::backend::DeviceMemPointer_CL<float> *> skepu_container_stencil, size_t elemPerPx, 
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *output, skepu2::backend::DeviceMemPointer_CL<unsigned char> *wrap,
		size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, unsigned char pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_COL);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), std::get<1>(skepu_container_stencil)->getDeviceDataPointer(), std::get<0>(skepu_container_stencil)->size(), elemPerPx,  output->getDeviceDataPointer(),
			wrap->getDeviceDataPointer(), n, overlap, out_offset, out_numelements, poly, pad, blocksPerCol, rowWidth, colWidth);
		clSetKernelArg(kernel, 3 + 12, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix col-wise kernel");
	}
	
	static void mapOverlapMatrixColWiseMulti
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *input, std::tuple<skepu2::Vector<float> *, skepu2::backend::DeviceMemPointer_CL<float> *> skepu_container_stencil, size_t elemPerPx, 
		skepu2::backend::DeviceMemPointer_CL<unsigned char> *output, skepu2::backend::DeviceMemPointer_CL<unsigned char> *wrap,
		size_t n, size_t overlap, size_t in_offset, size_t out_numelements, int poly, int deviceType, unsigned char pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_COL_MULTI);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), std::get<1>(skepu_container_stencil)->getDeviceDataPointer(), std::get<0>(skepu_container_stencil)->size(), elemPerPx,  output->getDeviceDataPointer(),
			wrap->getDeviceDataPointer(), n, overlap, in_offset, out_numelements, poly, deviceType, pad, blocksPerCol, rowWidth, colWidth);
		clSetKernelArg(kernel, 3 + 13, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix col-wise multi kernel");
	}
};
