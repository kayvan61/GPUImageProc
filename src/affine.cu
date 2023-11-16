#include "cuda.h"
#include <iostream>
#include <cstdio>

#include "Image.hpp"

__constant__ float transform[9];

__global__ void move(Pixel* img_out, Pixel* outLocs, Pixel *img_in, float* trans, unsigned image_w, unsigned image_h) {
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = 0, y = 0;
    x += img_x * trans[0];
    x += img_y * trans[1];
    x += 1 * trans[2];

    y += img_x * trans[3];
    y += img_y * trans[4];
    y += 1 * trans[5];
    
    if(img_x >= image_w || img_y >= image_h)
        return;
    Pixel srcPix = img_in[img_y * image_w + img_x];
    if(y < 0)
        return;
    if(y >= image_h)
        return;
    if(x < 0)
        return;
    if(x >= image_w)
        return;
    __syncthreads();
    img_out[y * image_w + x] = srcPix;
}

__global__ void newLocs(Pixel* outLocs, unsigned image_w, unsigned image_h, float* transform) {
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(img_x >= image_w || img_y >= image_h)
        return;
    outLocs[img_y * image_w + img_x] = {0, 0, 0};
    outLocs[img_y * image_w + img_x].r += img_x * transform[0];
    outLocs[img_y * image_w + img_x].r += img_y * transform[1];
    outLocs[img_y * image_w + img_x].r += 1 * transform[2];

    outLocs[img_y * image_w + img_x].g += img_x * transform[3];
    outLocs[img_y * image_w + img_x].g += img_y * transform[4];
    outLocs[img_y * image_w + img_x].g += 1 * transform[5];

    outLocs[img_y * image_w + img_x].b += img_x * transform[6];
    outLocs[img_y * image_w + img_x].b += img_y * transform[7];
    outLocs[img_y * image_w + img_x].b += 1 * transform[8];
}

int main(int argc, char** argv) {
    if(argc != 3) {
	  std::cout << "use: ./main <image in> <image out>" << std::endl;
	}

    Image img(argv[1]);
    Image locMap;
    Image result;
    std::pair<unsigned, unsigned> imgDims;
    imgDims = img.getImageDims();
    unsigned w = imgDims.first;
    unsigned h = imgDims.second;

    // float tranform[] = {
    //     0.707106f, 0.707106f, 0,
    //     -0.707106f, 0.707106f, 0,
    //     0, 0, 1
    // };
    float tranformHost[] = {
        -1, 0, (float)w,
        0, 1, 0,
        0, 0, 1
    };
    cudaMemcpyToSymbol(transform, tranformHost, sizeof(float) * 9);
    float* devTrans;
    cudaMalloc((void**)&devTrans, sizeof(float) * 9);
    cudaMemcpy(devTrans, tranformHost, sizeof(float) * 9, cudaMemcpyHostToDevice);
    
    
    printf("image dims: %d %d\n", w, h);
    locMap.createBlankDeviceImage(w, h, std::max(w, h) + 1);
    result.createBlankDeviceImage(w, h, 255);

    dim3 pictureTBDim = {32, 32};
    dim3 blocksTBDim = {(w/pictureTBDim.x) + 1, (h/pictureTBDim.y) + 1};

    img.copyToDevice();

    printf("thread dims: x: %d y: %d\n", pictureTBDim.x, pictureTBDim.y);
    printf("block  dims: x: %d y: %d\n", blocksTBDim.x, blocksTBDim.y);
    move<<<blocksTBDim, pictureTBDim>>>(result.getRawDeviceBuffer(), locMap.getRawDeviceBuffer(), img.getRawDeviceBuffer(), devTrans, w, h);

    cudaDeviceSynchronize();

    result.copyToHost();
    result.writeImage("rev.ppm");

    return 0;
}
