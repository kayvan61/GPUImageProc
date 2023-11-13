#include "cuda.h"
#include <iostream>
#include <cstdio>

#include "Image.hpp"

__global__ void conv(Pixel **img_out, Pixel **img_in, unsigned image_w, unsigned image_h,
                     float *conv_mask, unsigned conv_w, unsigned conv_h) {
    
    unsigned img_x = blockIdx.x * 32 + threadIdx.x;
    unsigned img_y = blockIdx.y * 32 + threadIdx.y;

    int conv_window_x_s = img_x - (conv_w/2);
    int conv_window_y_s = img_y - (conv_h/2);

    if(img_x < image_w && img_y < image_h) {

        Pixel resImage = {0, 0, 0};
        for(unsigned i = 0; i < conv_h; i++) {
            for(unsigned j = 0; j < conv_w; j++) {
                int conv_w_y = conv_window_y_s + i;
                if(conv_w_y < 0)
                    conv_w_y += image_h;
                if(conv_w_y >= image_h)
                    conv_w_y -= image_h;

                int conv_w_x = conv_window_x_s + j;
                if(conv_w_x < 0)
                    conv_w_x += image_w;
                if(conv_w_x >= image_w)
                    conv_w_x -= image_w;
                

                resImage.r += img_in[conv_w_y][conv_w_x].r * conv_mask[i*conv_w + j];
                resImage.g += img_in[conv_w_y][conv_w_x].g * conv_mask[i*conv_w + j];
                resImage.b += img_in[conv_w_y][conv_w_x].b * conv_mask[i*conv_w + j];
            }
        }
        img_out[img_y][img_x] = resImage;

    }
}

__global__ void clearImage(Pixel **a, unsigned w, unsigned h) {
    unsigned x = blockIdx.x * 32 + threadIdx.x;
    unsigned y = blockIdx.y * 32 + threadIdx.y;
    if(x < w && y < h) {
        a[y][x] = {0, 0, 0};
    }
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

int main() {
    float host_blur_host[] = {
        0.0625f, 0.125f, 0.0625f,
        0.125f, 0.25f, 0.125f,
        0.0625f, 0.125f, 0.0625f
    };
    // float host_blur_host[] = {
    //     0.0f, 0.0f, 0.0f,
    //     0.0f, 1.0f, 0.0f,
    //     0.0f, 0.0f, 0.0f
    // };
    float *dev_blur_mask;
    cudaMalloc((void**)&dev_blur_mask, sizeof(float) * 9);
    cudaMemcpy(dev_blur_mask, host_blur_host, sizeof(float) * 9, ::cudaMemcpyHostToDevice);


    Image img("TestImages/3c7re350fr861.ppm");
    Image outputImage;

    std::pair<unsigned, unsigned> imgDims;
    imgDims = img.getImageDims();
    unsigned w = imgDims.first;
    unsigned h = imgDims.second;
    printf("image dims: %d %d\n", w, h);
    outputImage.createBlankDeviceImage(w, h, 255);

    Pixel** raw_im = img.getRawHostBuffer();

    img.writeImage("TestImages/before.ppm");
    img.copyToDevice();

    if(w * h <= 1024) {
        dim3 pictureTBDim = {w, h};
        conv<<<1,pictureTBDim>>>(outputImage.getRawDeviceBuffer(), img.getRawDeviceBuffer(), w, h,
                                 dev_blur_mask, 3, 3);
    }
    else {
        dim3 pictureTBDim = {32, 32};
        dim3 blocksTBDim = {(w/32) + 1, (h/32) + 1};
        printf("thread dims: x: %d y: %d\n", 32, 32);
        printf("block  dims: x: %d y: %d\n", blocksTBDim.x, blocksTBDim.y);
        conv<<<blocksTBDim,pictureTBDim>>>(outputImage.getRawDeviceBuffer(), img.getRawDeviceBuffer(), w, h,
                                           dev_blur_mask, 3, 3);
    }

    cudaDeviceSynchronize();
    outputImage.copyToHost();
    outputImage.writeImage("TestImages/after.ppm");

    return 0;
}