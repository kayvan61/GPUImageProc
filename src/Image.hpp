#include <string>

typedef struct {
    int r, g, b;
} Pixel;

class Image {
    int w, h;
    int maxVal;
    Pixel **hostImage;
    Pixel **devImage;

    void freeHostImage();
    void freeDevImage();

public:
    ~Image();
    Image();
    Image(std::string fname);
    void readImage(std::string fname);
    void writeImage(std::string fname);
    void copyToDevice();

    Pixel** getRawDeviceBuffer();
    Pixel** getRawHostBuffer();
};
