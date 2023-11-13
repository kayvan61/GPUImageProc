#include "cuda.h"
#include "Image.hpp"
#include <iostream>
#include <cstdio>
#include <cassert>

Image::Image() {
    w = 0;
    h = 0;
    maxVal = 0;
    hostImage = nullptr;
    devImage = nullptr;
}

Image::Image(std::string fname) {
    w = 0;
    h = 0;
    maxVal = 0;
    hostImage = nullptr;
    devImage = nullptr;

    readHostImage(fname);
}

void Image::freeDevImage() {
    assert(devImage && "trying to free uninited devImage");

    Pixel **temp = (Pixel **)malloc(sizeof(Pixel*) * h);
    cudaMemcpy(temp, devImage, sizeof(Pixel*) * h, ::cudaMemcpyDeviceToHost);
    for(int i = 0; i < h; i++) {
        cudaFree(temp[i]);
    }
    free(temp);
    cudaFree(devImage);
    devImage = nullptr;
}

void Image::freeHostImage() {
    assert(hostImage && "trying to free uninited hostImage");

    for(int i = 0; i < h; i++) {
        free((hostImage[i]));
    }
    free(hostImage);
    hostImage = nullptr;
}

Image::~Image() {
    if(hostImage)
        freeHostImage();
    if(devImage)
        freeDevImage();
}

void Image::readHostImage(std::string fname) {
    FILE *in = fopen(fname.c_str(), "rb");
    fscanf(in, "P6 %d %d %d\n", &w, &h, &maxVal);

    allocHostImage();

    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            Pixel *curPix = &(hostImage[i][j]);
            curPix->r = fgetc(in);
            curPix->g = fgetc(in);
            curPix->b = fgetc(in);
        }
    }
    fclose(in);
}

Pixel** Image::getRawDeviceBuffer() {
    return devImage;
}
Pixel** Image::getRawHostBuffer() {
    return hostImage;
}

void Image::writeImage(std::string fname) {
    FILE *out = fopen(fname.c_str(), "wb");
    fprintf(out, "P6 %d %d %d\n", w, h, maxVal);
    for(unsigned i=0; i<h; i++) {
        for(unsigned j=0; j<w; j++) {
            Pixel *curPix = &(hostImage[i][j]);
            fputc(curPix->r, out);
            fputc(curPix->g, out);
            fputc(curPix->b, out);
        }
    }
    fclose(out);
}

void Image::allocDeviceImage() {
    cudaMalloc((void**)&devImage, sizeof(Pixel*) * h);
    Pixel **temp = (Pixel **)malloc(sizeof(Pixel*) * h);
    for(unsigned i = 0; i < h; i++) {
        cudaMalloc((void**)(&temp[i]), sizeof(Pixel) * w);
    }
    cudaMemcpy(devImage, temp, sizeof(Pixel*) * h, ::cudaMemcpyHostToDevice);
    free(temp);
    cudaDeviceSynchronize();
}
void Image::allocHostImage() {
    hostImage = (Pixel**)malloc(h * sizeof(Pixel*));

    for(unsigned i = 0; i < h; i++) {
        hostImage[i] = (Pixel*)malloc(w * sizeof(Pixel));
    }
}

void Image::copyToHost() {
    if(!hostImage) {
        allocHostImage();
    }
    Pixel **temp = (Pixel **)malloc(sizeof(Pixel*) * h);
    cudaMemcpy(temp, devImage, sizeof(Pixel*) * h, ::cudaMemcpyDeviceToHost);
    for(unsigned i = 0; i < h; i++) {
        cudaMemcpy(hostImage[i], temp[i], sizeof(Pixel) * w, ::cudaMemcpyDeviceToHost);
    }
    free(temp);
}

void Image::createBlankDeviceImage(int w, int h, int max) {
    this->w = w;
    this->h = h;
    maxVal = max;
    if(hostImage) 
        freeHostImage();
    if(devImage)
        freeDevImage();
    allocDeviceImage();
    cudaDeviceSynchronize();
}


void Image::copyToDevice() {
    if (!devImage)
        allocDeviceImage();
    
    // array of device pointers
    Pixel **temp = (Pixel **)malloc(sizeof(Pixel*) * h);
    cudaMemcpy(temp, devImage, sizeof(Pixel*) * h, ::cudaMemcpyDeviceToHost);
    for(unsigned i = 0; i < h; i++) {
        cudaMemcpy(temp[i], hostImage[i], sizeof(Pixel) * w, ::cudaMemcpyHostToDevice);
    }
    free(temp);
    cudaDeviceSynchronize();
}

std::pair<unsigned, unsigned> Image::getImageDims() {
    return {w, h};
}