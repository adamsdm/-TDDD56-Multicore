/*! \file matrix_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions of the Matrix class.
 */

#ifdef SKEPU_OPENCL

namespace skepu2
{
	template<typename T>
	std::vector<std::pair<cl_kernel, backend::Device_CL*>> Matrix<T>::transposeKernels_CL;
	
	/*!
	 *  \brief Update device with matrix content.
	 *
	 *  Update device with a Matrix range by specifying rowsize and column size. This allows to create rowwise paritions.
	 *  If Matrix does not have an allocation on the device for
	 *  the current range, create a new allocation and if specified, also copy Matrix data to device.
	 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so matrix can keep track of where
	 *  and what it has stored on devices.
	 *
	 *  \param start Pointer to first element in range to be updated with device.
	 *  \param rows Number of rows.
	 *  \param cols Number of columns.
	 *  \param device Pointer to the device that should be synched with.
	 *  \param copy Boolean value that tells whether to only allocate or also copy matrix data to device. True copies, False only allocates.
	 */
	template <typename T>
	typename Matrix<T>::device_pointer_type_cl Matrix<T>::updateDevice_CL(T* start, size_type rows, size_type cols, backend::Device_CL* device, bool copy)
	{
		DEBUG_TEXT_LEVEL3("Matrix updating device OpenCL\n")
		
		std::pair<cl_device_id, std::pair<const T*, size_type>> key(device->getDeviceID(), std::pair<const T*, size_type>(start, rows * cols));
		auto result = m_deviceMemPointers_CL.find(key);
		
		if (result == m_deviceMemPointers_CL.end()) //insert new, alloc mem and copy
		{
			auto temp = new backend::DeviceMemPointer_CL<T>(start, rows * cols, device);
			if(copy)
			{
				// Make sure uptodate
				updateHost_CL();
				// Copy
				temp->copyHostToDevice();
			}
			result = m_deviceMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key,temp));
		}
		else if (copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
		{
			// Make sure uptodate
			updateHost_CL(); // FIX IT: Only check for this copy and not for all copies.
			// Copy
			result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
		}
		return result->second;
	}
	
	
	template <typename T>
	typename Matrix<T>::device_const_pointer_type_cl Matrix<T>::updateDevice_CL(const T* start, size_type rows, size_type cols, backend::Device_CL* device, bool copy) const
	{
		DEBUG_TEXT_LEVEL3("Matrix updating device OpenCL\n")
			
		std::pair<cl_device_id, std::pair<const T*, size_type>> key(device->getDeviceID(), std::pair<const T*, size_type>(start, rows * cols));
		auto result = m_deviceMemPointers_CL.find(key);
		
		if (result == this->m_deviceConstMemPointers_CL.end()) //insert new, alloc mem and copy
		{
			auto temp = new device_const_pointer_type_cl(start, rows * cols, device);
			if (copy)
				temp->copyHostToDevice();
			
			result = this->m_deviceConstMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key, temp));
		}
		else if (copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
		{
			result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
		}
		return result->second;
	}
	
	
	/*!
	 *  \brief Update device with matrix content.
	 *
	 *  Update device with a Matrix range by specifying rowsize only as number of rows is assumed to be 1 in this case.
	 *  Helper function, useful for scenarios where matrix need to be treated like Vector 1D.
	 *  If Matrix does not have an allocation on the device for
	 *  the current range, create a new allocation and if specified, also copy Matrix data to device.
	 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so matrix can keep track of where
	 *  and what it has stored on devices.
	 *
	 *  \param start Pointer to first element in range to be updated with device.
	 *  \param cols Number of columns.
	 *  \param device Pointer to the device that should be synched with.
	 *  \param copy Boolean value that tells whether to only allocate or also copy matrix data to device. True copies, False only allocates.
	 */
	template <typename T>
	typename Matrix<T>::device_pointer_type_cl Matrix<T>::updateDevice_CL(T* start, size_type cols, backend::Device_CL* device, bool copy)
	{
		return updateDevice_CL(start, (size_type)1, cols, device, copy);
	}
	
	template <typename T>
	typename Matrix<T>::device_const_pointer_type_cl Matrix<T>::updateDevice_CL(const T* start, size_type cols, backend::Device_CL* device, bool copy) const
	{
		return updateDevice_CL(start, (size_type)1, cols, device, copy);
	}
	
	/*!
	 *  \brief Flushes the matrix.
	 *
	 *  First it updates the matrix from all its device allocations, then it releases all allocations.
	 */
	template <typename T>
	void Matrix<T>::flush_CL()
	{
		DEBUG_TEXT_LEVEL3("Matrix flush OpenCL\n")
		
		updateHost_CL();
		releaseDeviceAllocations_CL();
	}

	/*!
	 *  \brief Updates the host from devices.
	 *
	 *  Updates the matrix from all its device allocations.
	 */
	template <typename T>
	inline void Matrix<T>::updateHost_CL() const
	{
		DEBUG_TEXT_LEVEL3("Matrix updating host OpenCL\n")
		
		if (!this->m_deviceMemPointers_CL.empty())
			for (auto &memptr : this->m_deviceMemPointers_CL)
				memptr.second->copyDeviceToHost();
	}
	
	
	/*!
	 *  \brief Invalidates the device data.
	 *
	 *  Invalidates the device data by releasing all allocations. This way the matrix is updated
	 *  and then data must be copied back to devices if used again.
	 */
	template<typename T>
	inline void Matrix<T>::invalidateDeviceData_CL() const
	{
		DEBUG_TEXT_LEVEL3("Matrix invalidating device data OpenCL\n")
		
		//deallocs all device mem for matrix for now
		if (!this->m_deviceMemPointers_CL.empty())
			releaseDeviceAllocations_CL();
		//Could maybe be made better by only setting a flag that data is not valid
	}
	
	
	/*!
	 *  \brief Releases device allocations.
	 *
	 *  Releases all device allocations for this matrix. The memory pointers are removed.
	 */
	template<typename T>
	inline void Matrix<T>::releaseDeviceAllocations_CL() const
	{
		DEBUG_TEXT_LEVEL3("Matrix releasing device allocations OpenCL\n")
		
		for (auto &memptr : this->m_deviceMemPointers_CL)
			delete memptr.second;
		
		this->m_deviceMemPointers_CL.clear();
	}
	
	
	/*!
	 *  A function called by the constructor. It creates the OpenCL program for the Matrix Transpose and saves a handle for
	 *  the kernel. The program is built from a string containing the above mentioned generic transpose kernel. The type
	 *  and function names in the generic kernel are relpaced by specific code before it is compiled by the OpenCL JIT compiler.
	 *
	 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
	 *  are used.
	 */
	template<typename T>
	inline void Matrix<T>::createOpenCLProgramForMatrixTranspose()
	{
		// This function is idempotent
		if (Matrix<T>::transposeKernels_CL.size() > 0)
			return;
		
		// OpenCL Transpose kernel. Modified the transpose kernel provided by NVIDIA to make it work for any problem size rather than just perfect size such as 1024X1024.
		static const std::string TransposeKernelNoBankConflicts_CL = R"~~~(
		__kernel void transposeNoBankConflicts(__global TYPE* odata, __global TYPE* idata, size_t width, size_t height, __local TYPE* sdata)
		{
			int xIndex = get_group_id(0) * TILE_DIM + get_local_id(0);
			int yIndex = get_group_id(1) * TILE_DIM + get_local_id(1);
			int index_in = xIndex + (yIndex) * width;
			
			if (xIndex < width && yIndex < height)
				sdata[get_local_id(1) * TILE_DIM + get_local_id(0)] = idata[index_in];
			
			xIndex = get_group_id(1) * TILE_DIM + get_local_id(0);
			yIndex = get_group_id(0) * TILE_DIM + get_local_id(1);
			int index_out = xIndex + (yIndex) * height;
			
			barrier(CLK_LOCAL_MEM_FENCE);
			
			if (xIndex < height && yIndex < width)
				odata[index_out] = sdata[get_local_id(0) * TILE_DIM + get_local_id(1)];
		}
		)~~~";
		
		std::string kernelName = "transposeNoBankConflicts";
		std::string datatype_CL = getDataTypeCL<T>();
		std::stringstream totalSource;
		
		if (datatype_CL == "double")
			totalSource << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
		
		totalSource << backend::cl_helpers::replaceSizeT(TransposeKernelNoBankConflicts_CL);
		
		std::stringstream buildOptions;
		buildOptions << "-DTYPE=\"" << datatype_CL << "\" -DTILE_DIM=" << TILE_DIM;
		
		DEBUG_TEXT_LEVEL3("Transpose kernel source:\n" << totalSource.str());
		DEBUG_TEXT_LEVEL3("Transpose kernel build options:\n" << buildOptions.str());
		
		// Builds the code and creates kernel for all devices
		for (backend::Device_CL *device : backend::Environment<T>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = backend::cl_helpers::buildProgram(device, totalSource.str(), buildOptions.str());
			cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
			CL_CHECK_ERROR(err, "Error creating kernel '" << kernelName << "':" << err);
			Matrix<T>::transposeKernels_CL.emplace_back(kernel, device);
		}
	}
	
}

#endif
