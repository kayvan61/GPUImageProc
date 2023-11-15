#include "cuda.h"
#include <iostream>
#include <cstdio>

#include "Image.hpp"

__global__ void move(Pixel* img_out, Pixel* outLocs, Pixel *img_in, unsigned image_w, unsigned image_h) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(img_x >= image_w || img_y >= image_h)
        return;
    
    Pixel in_pix = img_in[img_y * image_w + img_x];
    int out_x = outLocs[img_y * image_w + img_x].r;
    int out_y = outLocs[img_y * image_w + img_x].g;
    while(out_y < 0)
        out_y += image_h;
    while(out_y >= image_h)
        out_y -= image_h;

    while(out_x < 0)
        out_x += image_w;
    while(out_x >= image_w)
        out_x -= image_w;
    img_out[out_y * image_w + out_x] = in_pix;
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

    // float tranform[] = {
    //     0.707106f, 0.707106f, 0,
    //     -0.707106f, 0.707106f, 0,
    //     0, 0, 1
    // };
    float tranform[] = {
        -1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    float* devTrans;
    cudaMalloc((void**)&devTrans, sizeof(float) * 9);
    cudaMemcpy(devTrans, tranform, sizeof(float) * 9, cudaMemcpyHostToDevice);
    
    std::pair<unsigned, unsigned> imgDims;
    imgDims = img.getImageDims();
    unsigned w = imgDims.first;
    unsigned h = imgDims.second;
    printf("image dims: %d %d\n", w, h);
    locMap.createBlankDeviceImage(w, h, std::max(w, h) + 1);
    result.createBlankDeviceImage(w, h, 255);

    dim3 pictureTBDim = {32, 32};
    dim3 blocksTBDim = {(w/32) + 1, (h/32) + 1};

    img.copyToDevice();

    printf("thread dims: x: %d y: %d\n", pictureTBDim.x, pictureTBDim.y);
    printf("block  dims: x: %d y: %d\n", blocksTBDim.x, blocksTBDim.y);
    newLocs<<<blocksTBDim,pictureTBDim>>>(locMap.getRawDeviceBuffer(), w, h, devTrans);
    cudaDeviceSynchronize();
    move<<<blocksTBDim, pictureTBDim>>>(result.getRawDeviceBuffer(), locMap.getRawDeviceBuffer(), img.getRawDeviceBuffer(), w, h);

    cudaDeviceSynchronize();

    result.copyToHost();
    result.writeImage("rev.ppm");

    return 0;
}
