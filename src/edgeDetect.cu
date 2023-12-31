#include "cuda.h"
#include <iostream>
#include <cstdio>

#include "Image.hpp"

#define COLOR_VALS 256
#define ONE_HIST_SIZE  COLOR_VALS
#define BLOCKSIZE 1024

__global__ void sobelSum(float *intentOut, float *dirOut, float *imX, float *imY, unsigned w, unsigned h) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(img_x < w && img_y < h) {
        intentOut[img_y * w + img_x] = sqrt(imX[img_y * w + img_x] * imX[img_y * w + img_x] + 
                                       imY[img_y * w + img_x] * imY[img_y * w + img_x]);
        dirOut[img_y * w + img_x]    = (57.29577951 * atan2(imY[img_y * w + img_x] * imY[img_y * w + img_x], 
                                             imX[img_y * w + img_x] * imX[img_y * w + img_x])) / 45;
    }
}

__global__ void dualThresh(float *out, float *in, float *low, float *high, unsigned w, unsigned h) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(img_x < w && img_y < h) {
        if(in[img_y * w + img_x] > 70)
            out[img_y * w + img_x] = 255;
        else if (in[img_y * w + img_x] < 20)
            out[img_y * w + img_x] = 0;
    }
}

__global__ void gradientDimin(float *imgOut, float* intens, float *dir, unsigned w, unsigned h) {
    int img_x = blockIdx.x * blockDim.x + threadIdx.x;
    int img_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(img_x+1 < w && img_y+1 < h  && img_x-1 >= 0 && img_y-1 >= 0) {
        float c = intens[img_y * w + img_x];
        float l, r;
        switch((int)dir[img_y * w + img_x]) {
            case 0: // left right
                l = intens[img_y * w + img_x - 1];
                r = intens[img_y * w + img_x + 1];
                imgOut[img_y * w + img_x] = l > c ? 0 : r > c ? 0 : c;
                break;
            case 3: // nw
                l = intens[(img_y+1) * w + img_x - 1];
                r = intens[(img_y-1) * w + img_x + 1];
                imgOut[img_y * w + img_x] = l > c ? 0 : r > c ? 0 : c;
                break;
            case 2: // north south
                l = intens[(img_y-1) * w + img_x - 1];
                r = intens[(img_y+1) * w + img_x + 1];
                imgOut[img_y * w + img_x] = l > c ? 0 : r > c ? 0 : c;
                break;
            case 1: // ne
                l = intens[(img_y - 1) * w + img_x - 1];
                r = intens[(img_y + 1) * w + img_x + 1];
                imgOut[img_y * w + img_x] = l > c ? 0 : r > c ? 0 : c;
                break;
        }
    }
}

// assumes flat threads
__global__ void max(float *img_out, int *a, unsigned len) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int portion[];
    portion[threadIdx.x] = a[index];
    int eleToRed = blockDim.x;
    __syncthreads();
    for(int i = 1; i < blockDim.x; i *= 2, eleToRed /= 2) {
        if(2*threadIdx.x+i < eleToRed) {
            int resIdx = threadIdx.x;
            int ele1 = -1.0f; 
            int ele2 = -1.0f; 
            if(2*threadIdx.x < len) {
                ele1 = portion[2*threadIdx.x];
            }
            if(2*threadIdx.x + i < len) {
                ele2 = portion[2*threadIdx.x+1];
            }
            portion[resIdx] = ele1 > ele2 ? ele1 : ele2;
        }
        __syncthreads();
    }
    img_out[blockIdx.x] = (float)portion[0];
}

__global__ void argMax(float *img_out, int *a, unsigned len) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    a[0] = 0;
    extern __shared__ int portion[];
    int *indexarr = portion + blockDim.x;
    if(threadIdx.x < len) {
        portion[threadIdx.x] = a[index];
        indexarr[threadIdx.x] = threadIdx.x;
    }
    else {
        portion[threadIdx.x] = -1;
        indexarr[threadIdx.x] = -1;
    }
    int eleToRed = blockDim.x;
    __syncthreads();
    for(int i = 1; i < blockDim.x; i *= 2, eleToRed /= 2) {
        if(2*threadIdx.x+i < eleToRed) {
            int resIdx = threadIdx.x;
            int ele1 = -1; 
            int idx1 = -1;
            int ele2 = -1; 
            int idx2 = -1;
            if(2*threadIdx.x < len) {
                ele1 = portion[2*threadIdx.x];
                idx1 = indexarr[2*threadIdx.x];
            }
            if(2*threadIdx.x + i < len) {
                ele2 = portion[2*threadIdx.x+1];
                idx2 = indexarr[2*threadIdx.x+1];
            }
            indexarr[resIdx] = ele1 > ele2 ? idx1 : idx2;
            portion[resIdx] = ele1 > ele2 ? ele1 : ele2;
        }
        __syncthreads();
    }
    img_out[blockIdx.x] = (float)indexarr[0];
}


__global__ void imgSum(float *img_out, float *a, float *b, unsigned image_w, unsigned image_h) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(img_x < image_w && img_y < image_h) {
        img_out[img_y * image_w + img_x] = a[img_y * image_w + img_x] + b[img_y * image_w + img_x];
    }
}

__global__ void toGrey(float *img_out, Pixel *img_in, unsigned image_w, unsigned image_h) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(img_x < image_w && img_y < image_h) {
        Pixel src = img_in[img_y * image_w + img_x];
        img_out[img_y * image_w + img_x] = (0.2126*src.r + 0.7152*src.g + 0.0722*src.b);
    }
}

__global__ void conv(float *img_out, float *img_in, unsigned image_w, unsigned image_h,
                     float *conv_mask, unsigned conv_w, unsigned conv_h) {
    
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    int conv_window_x_s = img_x - (conv_w/2);
    int conv_window_y_s = img_y - (conv_h/2);

    if(img_x < image_w && img_y < image_h) {

        float r = 0;
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
                
                r += img_in[conv_w_y * image_w + conv_w_x] * conv_mask[i*conv_w + j];
                __syncthreads();
            }
        }
        img_out[img_y * image_w + img_x] = r;

    }
}

__global__ void sharedConv(float *img_out, float *img_in, unsigned image_w, unsigned image_h,
                     float *conv_mask, unsigned conv_w, unsigned conv_h) {
    
    extern __shared__ float localImageData[];
    unsigned conv_w_pad = conv_w / 2;
    unsigned conv_h_pad = conv_h / 2;

    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int shMemSize = (blockDim.y) * (blockDim.x);

    if(img_x < image_w && img_y < image_h) {

        // move all the required data into the shared memory

        localImageData[(threadIdx.y * blockDim.x ) + threadIdx.x] = img_in[img_y * image_w + img_x];
        __syncthreads();

        float resImage = 0.0f;
        for(unsigned i = 0; i < conv_h; i++) {
            for(unsigned j = 0; j < conv_w; j++) {
                // local idx
                int shMemIdx = ((threadIdx.y - conv_h_pad + i) * blockDim.x) + threadIdx.x - conv_w_pad + j;
                if(shMemIdx < 0) {
                    shMemIdx = 0;
                }
                if(shMemIdx >= shMemSize) {
                    shMemIdx = shMemSize - 1;
                }
                float curPix = localImageData[shMemIdx];

                resImage += curPix * conv_mask[i*conv_w + j];
            }
        }
        img_out[img_y * image_w + img_x] = resImage;
    }
}

__global__ void reduce(int *histOut, int* localHistsIn) {
    unsigned res_idx = threadIdx.x;

    unsigned histogramIndex = (blockIdx.y * gridDim.x * ONE_HIST_SIZE) + (blockIdx.x * ONE_HIST_SIZE) + res_idx;

    int val = localHistsIn[histogramIndex];

    atomicAdd(&(histOut[res_idx]), val);
}

__global__ void histogram(int* localHistOut, float *img_in, unsigned image_w, unsigned image_h) {
    unsigned img_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned img_y = blockIdx.y;

    if(img_y >= image_h || img_x >= image_w) {
        return;
    }

    unsigned histogramIndex = (blockIdx.y * gridDim.x + blockIdx.x) * ONE_HIST_SIZE;
    unsigned histogramIndex_r = histogramIndex + (int)img_in[img_y * image_w + img_x];

    atomicAdd(&(localHistOut[histogramIndex_r]), 1);
}

int main(int argc, char** argv) {
    if(argc != 3) {
	  std::cout << "use: ./main <image in> <image out>" << std::endl;
	}


    int fliter_dim  =  5;
    int filter_size = fliter_dim * fliter_dim;
    float host_blur_host[] = {
	  0.0030f,    0.0133f,    0.0219f,    0.0133f,    0.0030f,
	  0.0133f,    0.0596f,    0.0983f,    0.0596f,    0.0133f,
	  0.0219f,    0.0983f,    0.1621f,    0.0983f,    0.0219f,
	  0.0133f,    0.0596f,    0.0983f,    0.0596f,    0.0133f,
	  0.0030f,    0.0133f,    0.0219f,    0.0133f,    0.0030f
    };
    int sobelDim = 3;
    int sobelSize = sobelDim * sobelDim;
    float hostIx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float hostIy[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    float *dev_blur_mask;
    cudaMalloc((void**)&dev_blur_mask, sizeof(float) * filter_size);
    cudaMemcpy(dev_blur_mask, host_blur_host, sizeof(float) * filter_size, ::cudaMemcpyHostToDevice);
    float *devIx, *devIy;
    cudaMalloc((void**)&devIx, sizeof(float) * sobelSize);
    cudaMalloc((void**)&devIy, sizeof(float) * sobelSize);
    cudaMemcpy(devIx, hostIx, sizeof(float) * sobelSize, ::cudaMemcpyHostToDevice);
    cudaMemcpy(devIy, hostIy, sizeof(float) * sobelSize, ::cudaMemcpyHostToDevice);


    Image img(argv[1]);

    std::pair<unsigned, unsigned> imgDims;
    imgDims = img.getImageDims();
    unsigned w = imgDims.first;
    unsigned h = imgDims.second;
    printf("image dims: %d %d\n", w, h);
    img.copyToDevice();

    Image outputImage(w, h);

    float* g_img;
    float* g_img2;
    float* g_img3;
    float* g_img4;
    float* g_imgsmooth;
    float *intens, *dir;
    cudaMalloc(&g_img, sizeof(float)*w*h);
    cudaMalloc(&g_img2, sizeof(float)*w*h);
    cudaMalloc(&g_img3, sizeof(float)*w*h);
    cudaMalloc(&g_img4, sizeof(float)*w*h);
    cudaMalloc(&g_imgsmooth, sizeof(float)*w*h);
    cudaMalloc(&intens, sizeof(float)*w*h);
    cudaMalloc(&dir, sizeof(float)*w*h);

    dim3 histTB = {BLOCKSIZE, 1};
    dim3 histGB = {(w/BLOCKSIZE) + 1, h};
    int* devHist;
    int* devLocalHists;
    float *low, *high;
    cudaMalloc((void**)&devLocalHists, sizeof(int) * histGB.x * histGB.y * ONE_HIST_SIZE);
    cudaMemset(devLocalHists, 0, sizeof(int) * histGB.x * histGB.y * ONE_HIST_SIZE);
    cudaMalloc((void**)&devHist, sizeof(int) * ONE_HIST_SIZE);
    cudaMemset(devHist, 0, sizeof(int) * ONE_HIST_SIZE);
    cudaMalloc((void**)&low, sizeof(float));
    cudaMalloc((void**)&high, sizeof(float));


    outputImage.createBlankDeviceImage(w, h, 255);

    dim3 pictureTBDim = {32,32};
    dim3 blocksTBDim = {(w/pictureTBDim.x) + 1, (h/pictureTBDim.y) + 1};
    printf("thread dims: x: %d y: %d\n", 32, 32);
    printf("block  dims: x: %d y: %d\n", blocksTBDim.x, blocksTBDim.y);
    int sharedMem = (pictureTBDim.x);
    sharedMem *= sharedMem;

    // as opt as it gets
    toGrey<<<{(w/128) + 1, h}, {128, 1}>>>(g_img, img.getRawDeviceBuffer(), w, h);
    // 
    sharedConv<<<blocksTBDim, pictureTBDim, sizeof(float) * (sharedMem)>>>(g_imgsmooth, g_img, w, h, dev_blur_mask, fliter_dim, fliter_dim);
    sharedConv<<<blocksTBDim, pictureTBDim, sizeof(float) * (sharedMem)>>>(g_img , g_imgsmooth, w, h, devIx, sobelDim, sobelDim);
    sharedConv<<<blocksTBDim, pictureTBDim, sizeof(float) * (sharedMem)>>>(g_img3, g_imgsmooth, w, h, devIy, sobelDim, sobelDim);
    sobelSum<<<blocksTBDim, pictureTBDim>>>(intens, dir, g_img, g_img3, w, h);
    gradientDimin<<<blocksTBDim, pictureTBDim>>>(g_img2, intens, dir, w, h);
    histogram<<<histGB,histTB>>>(devLocalHists, g_img2, w, h);
    reduce<<<histGB, ONE_HIST_SIZE>>>(devHist, devLocalHists);
    argMax<<<1, 1024, sizeof(int) * 1024 * 2>>>(low, devHist, ONE_HIST_SIZE/2);
    argMax<<<1, 1024, sizeof(int) * 1024 * 2>>>(high, devHist + ONE_HIST_SIZE/2, ONE_HIST_SIZE/2);
    dualThresh<<<blocksTBDim, pictureTBDim>>>(g_img4, g_img2, low, high, w, h);

    cudaDeviceSynchronize();

    int *hostHist = (int*)malloc(sizeof(int) * ONE_HIST_SIZE);
    cudaMemcpy(hostHist, devHist, sizeof(int) * ONE_HIST_SIZE, cudaMemcpyDeviceToHost);
    int sum = 0;
    printf("====== result ======\n");
    for(int i = 0; i < ONE_HIST_SIZE; i++) {
        sum += hostHist[i];
        printf("%d, ", hostHist[i]);
    }
    printf("\n");
    printf("sum: %d\n", sum);   


    float hostLow, hostHigh;
    cudaMemcpy(&hostLow, low, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostHigh, high, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "High " << hostHigh << std::endl;
    std::cout << "Low " << hostLow << std::endl;

    float* hostOut = (float*)malloc(sizeof(float) * w * h);
    cudaMemcpy(hostOut, intens, sizeof(float) * w * h, cudaMemcpyDeviceToHost);

    char fname[250];
    // sprintf(fname, "i-%s", argv[2]);

    // outputImage.setChannel(hostOut, "r");
    // outputImage.setChannel(hostOut, "g");
    // outputImage.setChannel(hostOut, "b");
    // outputImage.writeImage(fname);

    // cudaMemcpy(hostOut, g_img3, sizeof(float) * w * h, cudaMemcpyDeviceToHost);
    // sprintf(fname, "y-%s", argv[2]);
    // outputImage.setChannel(hostOut, "r");
    // outputImage.setChannel(hostOut, "g");
    // outputImage.setChannel(hostOut, "b");
    // outputImage.writeImage(fname);

    // cudaMemcpy(hostOut, g_imgsmooth, sizeof(float) * w * h, cudaMemcpyDeviceToHost);
    // sprintf(fname, "s-%s", argv[2]);
    // outputImage.setChannel(hostOut, "r");
    // outputImage.setChannel(hostOut, "g");
    // outputImage.setChannel(hostOut, "b");
    // outputImage.writeImage(fname);

    // cudaMemcpy(hostOut, g_img, sizeof(float) * w * h, cudaMemcpyDeviceToHost);
    // sprintf(fname, "x-%s", argv[2]);
    // outputImage.setChannel(hostOut, "r");
    // outputImage.setChannel(hostOut, "g");
    // outputImage.setChannel(hostOut, "b");
    // outputImage.writeImage(fname);

    // cudaMemcpy(hostOut, g_img2, sizeof(float) * w * h, cudaMemcpyDeviceToHost);
    // sprintf(fname, "d-%s", argv[2]);
    // outputImage.setChannel(hostOut, "r");
    // outputImage.setChannel(hostOut, "g");
    // outputImage.setChannel(hostOut, "b");
    // outputImage.writeImage(fname);

    // cudaMemcpy(hostOut, g_img4, sizeof(float) * w * h, cudaMemcpyDeviceToHost);
    // sprintf(fname, "%s", argv[2]);
    // outputImage.setChannel(hostOut, "r");
    // outputImage.setChannel(hostOut, "g");
    // outputImage.setChannel(hostOut, "b");
    // outputImage.writeImage(fname);

    return 0;
}

