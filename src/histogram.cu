#include "cuda.h"
#include <iostream>
#include <cstdio>
#include "scan.h"

#define FLT_EPSILON 1.19209290E-07F 
#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cerr << "CUDA Error: " << \
    cudaGetErrorString(XXX) << ", " << __FILE__ \
    << ":" << __LINE__ << std::endl; cudaDeviceSynchronize(); } while (0)

#include "Image.hpp"

#define SCAN_HISTO

#define COLOR_VALS 256
#define ONE_HIST_SIZE  COLOR_VALS * 3
#define BLOCKSIZE 1024

__global__ void reduce(int *histOut, int* localHistsIn) {
    unsigned res_idx = threadIdx.x;

    unsigned histogramIndex = (blockIdx.y * gridDim.x * ONE_HIST_SIZE) + (blockIdx.x * ONE_HIST_SIZE) + res_idx;

    int val = localHistsIn[histogramIndex];

    atomicAdd(&(histOut[res_idx]), val);
}

__global__ void maskVal(int *img_out, int *img_in, unsigned image_w, unsigned image_h, int val) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(img_x < image_w && img_y < image_h) {
        img_out[img_y * image_w + img_x] = img_in[img_y * image_w + img_x] == val  ? 1 : 0;
    }
}

__global__ void splitImage(int *r, int *g, int *b, Pixel *img_in, unsigned image_w, unsigned image_h) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.x + threadIdx.y;

    if(img_x < image_w && img_y < image_h) {
        r[img_y * image_w + img_x] = img_in[img_y * image_w + img_x].r;
        g[img_y * image_w + img_x] = img_in[img_y * image_w + img_x].g;
        b[img_y * image_w + img_x] = img_in[img_y * image_w + img_x].b;
    }
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
    img.copyToDevice();
    
    std::pair<unsigned, unsigned> imgDims;
    imgDims = img.getImageDims();
    unsigned w = imgDims.first;
    unsigned h = imgDims.second;
    printf("image dims: %d %d\n", w, h);


#ifdef SCAN_HISTO
    Image mask;
    mask.createBlankDeviceImage(w, h, 0);
    dim3 pictureTBDim = {32, 32};
    dim3 blocksTBDim = {(w/32) + 1, (h/32) + 1};

    int* devHist;
    cudaMalloc((void**)&devHist, sizeof(int) * ONE_HIST_SIZE);
    cudaMemset(devHist, 0, sizeof(int) * ONE_HIST_SIZE);
    int* devR, *sumR;
    int* devG, *sumG;
    int* devB, *sumB;
    CUDA_WARN(cudaMalloc((void**)&devR, sizeof(int) * nextPow2(w * h)));
    CUDA_WARN(cudaMalloc((void**)&devG, sizeof(int) * nextPow2(w * h)));
    CUDA_WARN(cudaMalloc((void**)&devB, sizeof(int) * nextPow2(w * h)));
    CUDA_WARN(cudaMalloc((void**)&sumR, sizeof(int) * nextPow2(w * h)));
    CUDA_WARN(cudaMalloc((void**)&sumG, sizeof(int) * nextPow2(w * h)));
    CUDA_WARN(cudaMalloc((void**)&sumB, sizeof(int) * nextPow2(w * h)));

    for(int i = 0; i < COLOR_VALS; i++) {
        splitImage<<<blocksTBDim, pictureTBDim>>>(sumR, sumG, sumB, img.getRawDeviceBuffer(), w, h);
        maskVal<<<blocksTBDim, pictureTBDim>>>(devR, sumR, w, h, i);
        maskVal<<<blocksTBDim, pictureTBDim>>>(devG, sumG, w, h, i);
        maskVal<<<blocksTBDim, pictureTBDim>>>(devB, sumB, w, h, i);
        exclusive_scan(devR, nextPow2(w * h), sumR);
        exclusive_scan(devG, nextPow2(w * h), sumG);
        exclusive_scan(devB, nextPow2(w * h), sumB);

        cudaDeviceSynchronize();

        CUDA_WARN(cudaMemcpy(&(devHist[3*i  ]), sumR + (w * h), sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_WARN(cudaMemcpy(&(devHist[3*i+1]), sumG + (w * h), sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_WARN(cudaMemcpy(&(devHist[3*i+2]), sumB + (w * h), sizeof(int), cudaMemcpyDeviceToDevice));
    }
#else
    dim3 pictureTBDim = {BLOCKSIZE, 1};
    dim3 blocksTBDim = {(w/BLOCKSIZE) + 1, h};

    int* devHist;
    cudaMalloc((void**)&devHist, sizeof(int) * ONE_HIST_SIZE);
    cudaMemset(devHist, 0, sizeof(int) * ONE_HIST_SIZE);

    int* devLocalHists;
    cudaMalloc((void**)&devLocalHists, sizeof(int) * blocksTBDim.x * blocksTBDim.y * ONE_HIST_SIZE);
    cudaMemset(devLocalHists, 0, sizeof(int) * blocksTBDim.x * blocksTBDim.y * ONE_HIST_SIZE);

    printf("thread dims: x: %d y: %d\n", pictureTBDim.x, pictureTBDim.y);
    printf("block  dims: x: %d y: %d\n", blocksTBDim.x, blocksTBDim.y);
    histogram<<<blocksTBDim,pictureTBDim>>>(devLocalHists, img.getRawDeviceBuffer(), w, h);
    reduce<<<blocksTBDim, ONE_HIST_SIZE>>>(devHist, devLocalHists);

    cudaDeviceSynchronize();
#endif

    int *host = (int*)malloc(sizeof(int) * ONE_HIST_SIZE);
    // cudaMemcpy(host, devHist, sizeof(int) * ONE_HIST_SIZE, cudaMemcpyDeviceToHost);
    // int sum = 0;
    // printf("====== result ======\n");
    // for(int i = 0; i < ONE_HIST_SIZE; i++) {
    //     sum += host[i];
    //     printf("%d ", host[i]);
    // }
    // for(int j = 0; j < h; j++) {
    //     for(int i = 0; i < w; i++) {
    //         Pixel ref = (img.getRawHostBuffer())[j][i];
    //         int r = ref.r;
    //         int g = ref.g;
    //         int b = ref.b;
    //         host[3*r    ] += 1;
    //         host[3*g + 1] += 1;
    //         host[3*b + 2] += 1;
    //     }
    // }
    printf("\n");
    //printf("sum: %d\n", sum);

    return 0;
}
