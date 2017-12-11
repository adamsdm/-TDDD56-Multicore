/*
 * Placeholder OpenCL kernel
 */


__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  const int sharedSize = 512;
  unsigned int index = get_global_id(0);                // Global thread idx
  unsigned int local_index = get_local_id(0);           // Local thread idx in work group
  unsigned const int local_size = get_local_size(0);    // Get no work items in work group
  __local int sharedMem[sharedSize];                    // Initialize a local memory for speedup with size 'local_size'

  sharedMem[local_index] = data[index]; // Write to shared memory
  barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE); // Sync work items

  
  for(unsigned int i=local_size/2; i>=1; i/=2){
    
    // Are we inside the work range?
    if(local_index < i){

      // Do the comparison
      if(sharedMem[local_index] < sharedMem[local_index + i]){
        sharedMem[local_index]=sharedMem[local_index+i];
      }
    }
  }

  // sharedMem[local_index] now contains the max number in the work group range
  // Write this back to the data at the global index position
  if(local_index == 0){
    data[index] = sharedMem[local_index];
    printf("%d\n", data[index]);

  }

  // Let global thread 0 find the max in each work group
  // Could be optimized by doing this on the CPU instead
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // Sync work items

  if(index==0){
    for(int i=0; i<= sharedSize; i+= local_size){
      if(data[i] > data[index])
        data[index] = data[i];
    }
  }
  
}


