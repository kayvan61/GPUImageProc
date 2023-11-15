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

Image::Image(int wid, int height) {
    w = wid;
    h = height;
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

void Image::setChannel(float* data, std::string channel) {
    if(!hostImage) {
        allocHostImage();
    }

    for(unsigned i=0; i<h; i++) {
        for(unsigned j=0; j<w; j++) {
            if(channel == "r")
                hostImage[i][j].r = data[i * w + j];
            if(channel == "g")
                hostImage[i][j].g = data[i * w + j];
            if(channel == "b")
                hostImage[i][j].b = data[i * w + j];
        }
    }
}

void Image::readHostImage(std::string fname) {
    FILE *in = fopen(fname.c_str(), "rb");
    fscanf(in, "P6 %d %d %d\n", &w, &h, &maxVal);

    allocHostImage();

    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            Pixel *curPix = &(hostImage[i][j]);
            curPix->r = (float)fgetc(in);
            curPix->g = (float)fgetc(in);
            curPix->b = (float)fgetc(in);
        }
    }
    fclose(in);
}

Pixel* Image::getRawDeviceBuffer() {
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
            fputc((int)curPix->r, out);
            fputc((int)curPix->g, out);
            fputc((int)curPix->b, out);
        }
    }
    fclose(out);
}

void Image::allocDeviceImage() {
    cudaMalloc((void**)&devImage, sizeof(Pixel) * h * w);
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
    for(unsigned i = 0; i < h; i++) {
        cudaMemcpy(hostImage[i], devImage+(i*w), sizeof(Pixel) * w, ::cudaMemcpyDeviceToHost);
    }
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
    
    for(unsigned i = 0; i < h; i++) {
        cudaMemcpy(devImage+(i*w), hostImage[i], sizeof(Pixel) * w, ::cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
}

std::pair<unsigned, unsigned> Image::getImageDims() {
    return {w, h};
}