/*
 * Placeholder OpenCL kernel
 */

__kernel void bitonic(__global unsigned int *data, const unsigned int length)
{ 
  unsigned int pos = 0;
  unsigned int val;

  //Something should happen here

  data[get_global_id(0)]=get_global_id(0);
}
