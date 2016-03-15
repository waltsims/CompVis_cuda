// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2015/2016, March 15 - April 15
// ###
// ###

#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

__device__ void add (  float *a, float *b, float *c,int n ){
	int ind = threadIdx.x + blockDim.x * blockIdx.x;
	if (ind < n) c[ind] = a[ind]  + b[ind];
}

__global__ void vecAdd ( float *a, float *b, float *c,int n ){
	add( a, b, c, n);
}

int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];
    for(int i=0; i<n; i++)
    {
        a[i] = i;
        b[i] = (i%5)+1;
        c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) c[i] = a[i] + b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) c[i] = 0;
    


    // GPU computation
    // ###
    // ### TODO: Implement the array addition on the GPU, store the result in "c"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "helper.h"
    
	float *d_a, *d_b, *d_c;
	cudaMalloc( &d_a, n * sizeof(float) );
	cudaMalloc( &d_b, n * sizeof(float) );
	cudaMalloc( &d_c, n * sizeof(float) );
	cudaMemcpy( d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( d_c, c, n * sizeof(float), cudaMemcpyHostToDevice);

	//Device Blocka allocation
	dim3 block = dim3(64, 1, 1); //64 threads
	// allocate blocks in grid
	dim3 grid = dim3( (n + block.x - 1 ) / block.x, 1, 1);
	
	vecAdd <<< grid, block >>> (d_a, d_b, d_c, n);
	
	cudaMemcpy ( c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost );

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    delete[] b;
    delete[] c;
}



