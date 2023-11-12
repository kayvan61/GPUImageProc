#include <string>

typedef struct {
    int r, g, b;
} Pixel;

class Image {
    int w, h;
    int maxVal;
    Pixel **rawImage;

public: 
    Image(std::string fname);


    void writeImage(std::string fname);
};
