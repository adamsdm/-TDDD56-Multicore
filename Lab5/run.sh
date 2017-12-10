rm a.out

g++ -c readppm.c milli.c
nvcc -c filter.cu
g++ filter.o readppm.o milli.o -lGL -lglut -L/usr/local/cuda/lib64 -lcudart
./a.out
