#include "cuda.h"
#include <iostream>

__global__ void populate(int *a) {
    printf("%d, %d, %d, %d, %d, %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
    int idx = blockIdx.x * 32 + threadIdx.x;
    a[idx] = idx;
}

int main() {
    int *retVal;
    int *a;
    retVal = (int*)malloc(sizeof(int) * 32 * 512);
    cudaMalloc((void**)&a, sizeof(int) * 32 * 512);
    dim3 threadBlockDim = {32};
    dim3 BlockDim = {512};
    populate<<<BlockDim, threadBlockDim>>>(a);
    cudaMemcpy(retVal, a, sizeof(int) * 32 * 512, ::cudaMemcpyDeviceToHost);
    for(int i = 512; i < 512 + 32; i++) {
        std::cout << retVal[i] << " ";
    }
    std::cout << "\n";
    return 0;
}