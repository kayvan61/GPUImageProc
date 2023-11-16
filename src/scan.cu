#include "scan.h"
#include "cuda.h"
#include <iostream>

#define BLOCKSIZE 512


__global__ void reduce(int* in_data, int length, int offset, int iter, int *out) {
  int gThId = blockIdx.x * blockDim.x  + threadIdx.x;
  
  int in_indx = (2*gThId+1) * offset - 1;
  int out_indx = (2*gThId+2) * offset - 1;
  if(gThId < iter) {
    in_data[out_indx] += in_data[in_indx];
  }
  //  __syncthreads();
}

__global__ void prep(int* in, int length, int* out) {
  int gThId = blockIdx.x * blockDim.x  + threadIdx.x;
  out[gThId] = gThId == length-1 ? 0 : in[gThId];
  //  __syncthreads();
}


// need the same partner as before
// but this time we swap, then sum
__global__ void down_sweep(int* in, int length, int* out, int offset, int iter) {
  int gThId = blockIdx.x * blockDim.x  + threadIdx.x;
  
  int in_indx = (2*gThId+1) * offset - 1;
  int out_indx = (2*gThId+2) * offset - 1;
  if(gThId < iter) {
    int t = out[in_indx];
    out[in_indx] = out[out_indx];
    out[out_indx] += t;
  }
  
  //  __syncthreads();
}

// This will do the scan for one block of threads
// next kernel will fix up the block sums!
__global__ void one_kernel_runner(int* in, int* out, int length, int* sums) {
  // do the whole scan in one kernel
  __shared__ int scratchPad[BLOCKSIZE * 2];
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int gStart = BLOCKSIZE * 2 * bid;

  // move to local
  scratchPad[2*tid] = in[gStart + 2*tid];
  scratchPad[2*tid+1] = in[gStart + 2*tid + 1];

  // Since we are now in the block we are going to work as if length is BLOCKSIZE*2
  int offset = 1;
  for (int i = BLOCKSIZE; i > 0; i >>= 1, offset <<= 1) {
    __syncthreads();
    int in_indx = (2*tid+1) * offset - 1;
    int out_indx = (2*tid+2) * offset - 1;
    if(tid < i) {
      scratchPad[out_indx] += scratchPad[in_indx];
    }
  }

  // prep by core 0 only
  if (threadIdx.x == 0) {
    sums[blockIdx.x] = scratchPad[(BLOCKSIZE * 2)-1];
    scratchPad[(BLOCKSIZE * 2)-1] = 0;
  }

  // now the downsweep
  for (int i = 1; i < BLOCKSIZE * 2; i <<= 1) {
    __syncthreads();
    offset >>= 1;
    int in_indx = (2*tid+1) * offset - 1;
    int out_indx = (2*tid+2) * offset - 1;
    if(tid < i) {
      int t = scratchPad[in_indx];
      scratchPad[in_indx] = scratchPad[out_indx];
      scratchPad[out_indx] += t;
    }
  }

  __syncthreads();
  out[gStart + 2*tid]     = scratchPad[2*tid];
  out[gStart + 2*tid + 1] = scratchPad[2*tid+1];
}

__global__ void phase_2_kernel(int* out, int length, int* sums){
  int gThId = blockDim.x * blockIdx.x + threadIdx.x;
  int bId   = (blockIdx.x/2);

  if(gThId < length) {
    // if(bThId == 0) {
    //   printf("bId %d, %d: %d\n", bId, gThId, sums[bId]);
    // }
    out[gThId] += sums[bId];
  }
}

// launch kernels to do a scan of list. pow 2 length. requires recursion
void scan_helper(int* list_to_scan, int* scan_out, int length){
  int num_blocks = max(nextPow2(length)/(2*BLOCKSIZE),1);
  int *sums;
  int *incs;
  cudaMalloc((void**)&sums, sizeof(int)*num_blocks*2);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("%d: CUDA error: %s\n", __LINE__, cudaGetErrorString(error));
    exit(-1);
  }
  incs = sums + num_blocks; // evil... too bad!
  one_kernel_runner<<<max(nextPow2(length)/(2*BLOCKSIZE),1), BLOCKSIZE>>>(list_to_scan, scan_out, nextPow2(length), sums);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("%d: CUDA error: %s\n", __LINE__, cudaGetErrorString(error));
    exit(-1);
  }


  if(num_blocks > 1) {
    scan_helper(sums, incs, num_blocks);
  }

  if(num_blocks > 1) {
    phase_2_kernel<<<max(nextPow2(length)/(BLOCKSIZE),1), BLOCKSIZE>>>(scan_out, nextPow2(length), incs);
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("%d: CUDA error: %s\n", __LINE__, cudaGetErrorString(error));
      exit(-1);
    }
  }
  cudaFree(sums);
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
    scan_helper(device_start, device_result, nextPow2(length));
}