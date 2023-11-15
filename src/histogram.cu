#include "cuda.h"
#include <iostream>
#include <cstdio>

#include "Image.hpp"

#define COLOR_VALS 256
#define ONE_HIST_SIZE  COLOR_VALS * 3
#define BLOCKSIZE 1024

__global__ void reduce(int *histOut, int* localHistsIn) {
    unsigned res_idx = threadIdx.x;

    unsigned histogramIndex = (blockIdx.y * gridDim.x * ONE_HIST_SIZE) + (blockIdx.x * ONE_HIST_SIZE) + res_idx;

    int val = localHistsIn[histogramIndex];

    atomicAdd(&(histOut[res_idx]), val);
}

__global__ void histogram(int* localHistOut, Pixel *img_in, unsigned image_w, unsigned image_h) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y;

    if(img_y >= image_h || img_x >= image_w) {
        return;
    }

    unsigned histogramIndex = (blockIdx.y * gridDim.x + blockIdx.x) * ONE_HIST_SIZE;
    unsigned histogramIndex_r = histogramIndex + (img_in[img_y * image_w + img_x].r * 3);
    unsigned histogramIndex_g = histogramIndex + (img_in[img_y * image_w + img_x].g * 3)+1;
    unsigned histogramIndex_b = histogramIndex + (img_in[img_y * image_w + img_x].b * 3)+2;

    atomicAdd(&(localHistOut[histogramIndex_r]), 1);
    atomicAdd(&(localHistOut[histogramIndex_g]), 1);
    atomicAdd(&(localHistOut[histogramIndex_b]), 1);

}

int main(int argc, char** argv) {
    if(argc != 2) {
	  std::cout << "use: ./main <image in>" << std::endl;
	}

    Image img(argv[1]);
    
    std::pair<unsigned, unsigned> imgDims;
    imgDims = img.getImageDims();
    unsigned w = imgDims.first;
    unsigned h = imgDims.second;
    printf("image dims: %d %d\n", w, h);

    dim3 pictureTBDim = {BLOCKSIZE, 1};
    dim3 blocksTBDim = {(w/BLOCKSIZE) + 1, h};

    int* devHist;
    int* devLocalHists;
    cudaMalloc((void**)&devLocalHists, sizeof(int) * blocksTBDim.x * blocksTBDim.y * ONE_HIST_SIZE);
    cudaMemset(devLocalHists, 0, sizeof(int) * blocksTBDim.x * blocksTBDim.y * ONE_HIST_SIZE);
    cudaMalloc((void**)&devHist, sizeof(int) * ONE_HIST_SIZE);
    cudaMemset(devHist, 0, sizeof(int) * ONE_HIST_SIZE);

    img.copyToDevice();

    printf("thread dims: x: %d y: %d\n", pictureTBDim.x, pictureTBDim.y);
    printf("block  dims: x: %d y: %d\n", blocksTBDim.x, blocksTBDim.y);
    histogram<<<blocksTBDim,pictureTBDim>>>(devLocalHists, img.getRawDeviceBuffer(), w, h);
    reduce<<<blocksTBDim, ONE_HIST_SIZE>>>(devHist, devLocalHists);

    cudaDeviceSynchronize();

    int *host = (int*)malloc(sizeof(int) * ONE_HIST_SIZE);
    cudaMemcpy(host, devHist, sizeof(int) * ONE_HIST_SIZE, cudaMemcpyDeviceToHost);
    int sum = 0;
    printf("====== result ======\n");
    for(int i = 0; i < ONE_HIST_SIZE; i++) {
        sum += host[i];
        printf("%d ", host[i]);
    }
    printf("\n");
    printf("sum: %d\n", sum);

    return 0;
}
