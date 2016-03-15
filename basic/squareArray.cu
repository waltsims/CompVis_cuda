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
#define CUDA_CHECK cuda_check(__FILE__, __LINE__)
void cuda_check(string file, int line) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    cout << endl
         << file << ", line " << line << ": " << cudaGetErrorString(e) << " ("
         << e << ")" << endl;
    exit(1);
  }
}

__device__ void square(float *a, int n) {
  int ind = threadIdx.x + blockDim.x * blockIdx.x;
  if (ind < n)
    a[ind] = a[ind] * a[ind];
}

__global__ void vecSq(float *a, int n) { square(a, n); }

int main(int argc, char **argv) {
  // alloc and init input arrays on host (CPU)
  int n = 10;
  float *a = new float[n];
  for (int i = 0; i < n; i++)
    a[i] = i;

  // CPU computation
  for (int i = 0; i < n; i++) {
    float val = a[i];
    val = val * val;
    a[i] = val;
  }

  // print result
  cout << "CPU:" << endl;
  for (int i = 0; i < n; i++)
    cout << i << ": " << a[i] << endl;
  cout << endl;

  // GPU computation
  // reinit data
  for (int i = 0; i < n; i++)
    a[i] = i;

  // ###
  // ### TODO: Implement the "square array" operation on the GPU and store the
  // result in "a"
  // ###
  // ### Notes:
  // ### 1. Remember to free all GPU arrays after the computation
  // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g.
  // "cudaMalloc(...); CUDA_CHECK;"
  // ###    For convenience this macro is defined directly in this file, later
  // we will only include "helper.h"

  // Memory allocation on Device
  float *d_a;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);

  // Device Blocka allocation
  dim3 block = dim3(64, 1, 1); // 64 threads
  // allocate blocks in grid
  dim3 grid = dim3((n + block.x - 1) / block.x, 1, 1);

  vecSq<<<grid, block>>>(d_a, n);

  cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);

  // print result
  cout << "GPU:" << endl;
  for (int i = 0; i < n; i++)
    cout << i << ": " << a[i] << endl;
  cout << endl;

  // free CPU arrays
  delete[] a;
}

