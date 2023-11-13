#include <string>

typedef struct {
    unsigned r, g, b;
} Pixel;

class Image {
    unsigned w, h;
    unsigned maxVal;
    Pixel **hostImage;
    Pixel **devImage;

    void freeHostImage();
    void freeDevImage();
    void allocDeviceImage();
    void allocHostImage();

public:
    ~Image();
    Image();
    Image(std::string fname);
    void createBlankDeviceImage(int w, int h, int max);
    void readHostImage(std::string fname);
    void writeImage(std::string fname);
    void copyToDevice();
    void copyToHost();
    std::pair<unsigned, unsigned> getImageDims();

    Pixel** getRawDeviceBuffer();
    Pixel** getRawHostBuffer();
};
