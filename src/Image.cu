#include "cuda.h"
#include "Image.hpp"
#include <iostream>
#include <cstdio>

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

    readImage(fname);
}

void Image::freeDevImage() {
    if(!devImage) 
        return;

    Pixel **temp = (Pixel **)malloc(sizeof(Pixel*) * h);
    cudaMemcpy(temp, devImage, sizeof(Pixel*) * h, ::cudaMemcpyDeviceToHost);
    for(int i = 0; i < h; i++) {
        cudaFree(temp[i]);
    }
    free(temp);
    cudaFree(devImage);
}

void Image::freeHostImage() {
    if(!hostImage) 
        return;
    for(int i = 0; i < h; i++) {
        free((hostImage[i]));
    }
    free(hostImage);
}

Image::~Image() {
    freeHostImage();
    freeDevImage();
}

void Image::readImage(std::string fname) {
    freeHostImage();

    devImage = nullptr;
    FILE *in = fopen(fname.c_str(), "rb");
    fscanf(in, "P6 %d %d %d\n", &w, &h, &maxVal);

    hostImage = (Pixel**)malloc(h * sizeof(Pixel*));

    for(int i = 0; i < h; i++) {
        hostImage[i] = (Pixel*)malloc(w * sizeof(Pixel));
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
    for(int i=0; i<h; i++) {
        for(int j=0; j<w; j++) {
            Pixel *curPix = &(hostImage[i][j]);
            fputc(curPix->r, out);
            fputc(curPix->g, out);
            fputc(curPix->b, out);
        }
    }
    fclose(out);
}

void Image::copyToDevice() {
    cudaMalloc((void**)&devImage, sizeof(Pixel*) * h);
    Pixel **temp = (Pixel **)malloc(sizeof(Pixel*) * h);
    for(int i = 0; i < h; i++) {
        cudaMalloc((void**)(&temp[i]), sizeof(Pixel) * w);
        cudaMemcpy(temp[i], hostImage[i], sizeof(Pixel) * w, ::cudaMemcpyHostToDevice);
    }
    cudaMemcpy(devImage, temp, sizeof(Pixel*) * h, ::cudaMemcpyHostToDevice);
    free(temp);
}