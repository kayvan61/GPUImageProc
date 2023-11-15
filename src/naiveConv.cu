#include "cuda.h"
#include <iostream>
#include <cstdio>

#include "Image.hpp"

__global__ void sharedConv(Pixel *img_out, Pixel *img_in, unsigned image_w, unsigned image_h,
                     float *conv_mask, unsigned conv_w, unsigned conv_h) {
    
    extern __shared__ Pixel localImageData[];
    unsigned conv_w_pad = conv_w / 2;
    unsigned conv_h_pad = conv_h / 2;

    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int shMemSize = (blockDim.y) * (blockDim.x);

    if(img_x < image_w && img_y < image_h) {

        // move all the required data into the shared memory

        localImageData[(threadIdx.y * blockDim.x ) + threadIdx.x] = img_in[img_y * image_w + img_x];
        __syncthreads();

        Pixel resImage = {0, 0, 0};
        for(unsigned i = 0; i < conv_h; i++) {
            for(unsigned j = 0; j < conv_w; j++) {
                // local idx
                int shMemIdx = ((threadIdx.y - conv_h_pad + i) * blockDim.x) + threadIdx.x - conv_w_pad + j;
                if(shMemIdx < 0 || shMemIdx >= shMemSize) {
                    continue;
                }
                Pixel& curPix = localImageData[shMemIdx];

                resImage.r += curPix.r * conv_mask[i*conv_w + j];
                resImage.g += curPix.g * conv_mask[i*conv_w + j];
                resImage.b += curPix.b * conv_mask[i*conv_w + j];
            }
        }
        img_out[img_y * image_w + img_x] = resImage;

    }
}

__global__ void conv(Pixel *img_out, Pixel *img_in, unsigned image_w, unsigned image_h,
                     float *conv_mask, unsigned conv_w, unsigned conv_h) {
    
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    int conv_window_x_s = img_x - (conv_w/2);
    int conv_window_y_s = img_y - (conv_h/2);

    if(img_x < image_w && img_y < image_h) {

        float r = 0, g = 0, b = 0;
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
                
                r += img_in[conv_w_y * image_w + conv_w_x].r * conv_mask[i*conv_w + j];
                g += img_in[conv_w_y * image_w + conv_w_x].g * conv_mask[i*conv_w + j];
                b += img_in[conv_w_y * image_w + conv_w_x].b * conv_mask[i*conv_w + j];
            }
        }
        img_out[img_y * image_w + img_x] = {r,g,b};

    }
}

__global__ void clearImage(Pixel *a, unsigned w, unsigned h) {
    unsigned x = blockIdx.x * 32 + threadIdx.x;
    unsigned y = blockIdx.y * 32 + threadIdx.y;
    if(x < w && y < h) {
        a[y * w + x] = {0, 0, 0};
    }
}


int main(int argc, char** argv) {
    if(argc != 3) {
	  std::cout << "use: ./main <image in> <image out>" << std::endl;
	}
    // float host_blur_host[] = {
	//   0.0030f,    0.0133f,    0.0219f,    0.0133f,    0.0030f,
	//   0.0133f,    0.0596f,    0.0983f,    0.0596f,    0.0133f,
	//   0.0219f,    0.0983f,    0.1621f,    0.0983f,    0.0219f,
	//   0.0133f,    0.0596f,    0.0983f,    0.0596f,    0.0133f,
	//   0.0030f,    0.0133f,    0.0219f,    0.0133f,    0.0030f

    // };
    const int filter_dim = 15;
    const int filter_size = filter_dim * filter_dim;
    float *host_blur_host = (float*)malloc(sizeof(float) * filter_size);
    for(int i = 0; i < filter_size; i++) {
        host_blur_host[i] = 1.0f/filter_size;
    }
    float *dev_blur_mask;
    cudaMalloc((void**)&dev_blur_mask, sizeof(float) * filter_size);
    cudaMemcpy(dev_blur_mask, host_blur_host, sizeof(float) * filter_size, ::cudaMemcpyHostToDevice);


    Image img(argv[1]);
    Image outputImage;

    std::pair<unsigned, unsigned> imgDims;
    imgDims = img.getImageDims();
    unsigned w = imgDims.first;
    unsigned h = imgDims.second;
    printf("image dims: %d %d\n", w, h);
    outputImage.createBlankDeviceImage(w, h, 255);

    Pixel** raw_im = img.getRawHostBuffer();

    img.copyToDevice();

    if(w * h <= 1024) {
        dim3 pictureTBDim = {w, h};
        conv<<<1,pictureTBDim>>>(outputImage.getRawDeviceBuffer(), img.getRawDeviceBuffer(), w, h,
                                 dev_blur_mask, filter_dim, filter_dim);
    }
    else {
        dim3 pictureTBDim = {32, 32};
        dim3 blocksTBDim = {(w/32) + 1, (h/32) + 1};
        printf("thread dims: x: %d y: %d\n", 32, 32);
        printf("block  dims: x: %d y: %d\n", blocksTBDim.x, blocksTBDim.y);
        //conv<<<blocksTBDim,pictureTBDim>>>(outputImage.getRawDeviceBuffer(), img.getRawDeviceBuffer(), w, h,
        //                                   dev_blur_mask, filter_dim, filter_dim);
        cudaDeviceSynchronize();
        outputImage.copyToHost();
        char outfname[255];
        sprintf(outfname, "slow-%s", argv[2]);
        std::cout << outfname << std::endl;
        outputImage.writeImage(outfname);

        dim3 newpictureTBDim = {32, 32};
        int sharedMem = (newpictureTBDim.x);
        sharedMem *= sharedMem;
        sharedConv<<<blocksTBDim, newpictureTBDim, sizeof(Pixel) * (sharedMem)>>>(outputImage.getRawDeviceBuffer(), img.getRawDeviceBuffer(), w, h,
                                           dev_blur_mask, filter_dim, filter_dim);
    }

    cudaDeviceSynchronize();
    outputImage.copyToHost();
    outputImage.writeImage(argv[2]);

    return 0;
}
