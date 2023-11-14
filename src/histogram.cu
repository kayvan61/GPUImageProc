#include "cuda.h"
#include <iostream>
#include <cstdio>

#include "Image.hpp"

#define COLOR_VALS 255
#define ONE_HIST_SIZE 255 * 3
#define BLOCKSIZE 1024

__global__ void reduce(int *histOut, int* localHistsIn) {
    unsigned res_idx = threadIdx.x;

    unsigned histogramIndex = (blockIdx.y * blockDim.x * ONE_HIST_SIZE) + (blockIdx.x * ONE_HIST_SIZE);
    unsigned histogramIndex_r = histogramIndex + res_idx;
    unsigned histogramIndex_g = histogramIndex + COLOR_VALS + res_idx;
    unsigned histogramIndex_b = histogramIndex + (2*COLOR_VALS) + res_idx;

    int val = localHistsIn[histogramIndex_r];

    atomicAdd(&(histOut[res_idx]), val);
}

__global__ void histogram(int* localHistOut, Pixel **img_in, unsigned image_w, unsigned image_h) {
    unsigned img_x = blockIdx.y * BLOCKSIZE + threadIdx.x;
    unsigned img_y = blockIdx.y;

    if(img_y >= image_h || img_x >= image_w) {
        return;
    }

    unsigned histogramIndex = (blockIdx.y * blockDim.x * ONE_HIST_SIZE) + (blockIdx.x * ONE_HIST_SIZE);
    unsigned histogramIndex_r = histogramIndex + img_in[img_y][img_x].r;
    unsigned histogramIndex_g = histogramIndex + COLOR_VALS + img_in[img_y][img_x].g;
    unsigned histogramIndex_b = histogramIndex + (2*COLOR_VALS) + img_in[img_y][img_x].b;

    atomicAdd(&(localHistOut[histogramIndex_r]), 1);
    atomicAdd(&(localHistOut[histogramIndex_g]), 1);
    atomicAdd(&(localHistOut[histogramIndex_b]), 1);
}

template<typename T>
void cuda2dAlloc(T*** outBuffer, unsigned w, unsigned h) {
    cudaMalloc(outBuffer, sizeof(T*) * h);
    T **temp = (T **)malloc(sizeof(T*) * h);
    for(unsigned i = 0; i < h; i++) {
        cudaMalloc((void**)(&temp[i]), sizeof(T) * w);
    }
    cudaMemcpy(*outBuffer, temp, sizeof(T*) * h, ::cudaMemcpyHostToDevice);
    free(temp);
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
    cudaMalloc((void**)&devHist, sizeof(int) * ONE_HIST_SIZE);
    cudaMemset(devHist, 0, sizeof(int) * ONE_HIST_SIZE);

    img.copyToDevice();

    printf("thread dims: x: %d y: %d\n", pictureTBDim.x, pictureTBDim.y);
    printf("block  dims: x: %d y: %d\n", blocksTBDim.x, blocksTBDim.y);
    histogram<<<blocksTBDim,pictureTBDim>>>(devLocalHists, img.getRawDeviceBuffer(), w, h);
    reduce<<<blocksTBDim, COLOR_VALS>>>(devHist, devLocalHists);

    cudaDeviceSynchronize();

    int *host = (int*)malloc(sizeof(int) * COLOR_VALS);
    cudaMemcpy(host, devHist, sizeof(int) * COLOR_VALS, cudaMemcpyDeviceToHost);
    for(int i = 0; i < COLOR_VALS; i++) {
        printf("%d ", host[i]);
    }
    printf("\n");

    return 0;
}
