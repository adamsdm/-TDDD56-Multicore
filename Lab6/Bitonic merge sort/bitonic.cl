/*
 * Placeholder OpenCL kernel
 */

inline void exchange(__global unsigned int *i, __global unsigned int *j){
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

__kernel void bitonic(__global unsigned int *data, const unsigned int N)
{ 
  unsigned int pos = 0;
  unsigned int val;
  unsigned int index = get_global_id(0);

  //Something should happen here

  unsigned int i,j,k;

  if(index == 0){
    for (k=2;k<=N;k=2*k) // Outer loop, double size for each step
    {
      for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
      {
        for (i=0;i<N;i++) // Loop over data
        {
          int ixj=i^j; // Calculate indexing!
          if ((ixj)>i)
          {
            if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
            if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
          }
        }
      }
    }

  }
}
