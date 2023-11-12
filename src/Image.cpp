#include "Image.hpp"
#include <iostream>
#include <cstdio>


Image::Image(std::string fname) {
    FILE *in = fopen(fname.c_str(), "rb");
    fscanf(in, "P6 %d %d %d\n", &w, &h, &maxVal);
    printf("P6 %d %d\n", w, h);

    rawImage = (Pixel**)malloc(h * sizeof(Pixel*));

    for(int i = 0; i < h; i++) {
        rawImage[i] = (Pixel*)malloc(w * sizeof(Pixel));
        for(int j = 0; j < w; j++) {
            Pixel *curPix = &(rawImage[i][j]);
            curPix->r = fgetc(in);
            curPix->g = fgetc(in);
            curPix->b = fgetc(in);
        }
    }

    fclose(in);
}

void Image::writeImage(std::string fname) {
    FILE *out = fopen(fname.c_str(), "wb");

    fprintf(out, "P6 %d %d %d\n", w, h, maxVal);
    for(int i=0; i<h; i++) {
        for(int j=0; j<w; j++) {
            Pixel *curPix = &(rawImage[i][j]);
            fputc(curPix->r, out);
            fputc(curPix->g, out);
            fputc(curPix->b, out);
        }
    }
    fclose(out);
}