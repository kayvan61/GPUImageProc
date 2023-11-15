CXX=nvcc

SRC_DIR := ./src
OBJ_DIR := ./objs
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
LDFLAGS := 
CPPFLAGS := 
CXXFLAGS := -g -O3 -ccbin=/opt/cuda/bin

all: conv histo affine edgeDetect

edgeDetect: ./objs/Image.o ./objs/edgeDetect.o
	$(CXX) $(LDFLAGS) -o $@ $^

affine: ./objs/Image.o ./objs/affine.o
	$(CXX) $(LDFLAGS) -o $@ $^

conv: ./objs/Image.o ./objs/naiveConv.o
	$(CXX) $(LDFLAGS) -o $@ $^

histo: ./objs/Image.o ./objs/histogram.o
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm $(OBJ_DIR)/*.o || true
	rm ./*.o || true
	rm histo || true
	rm conv || true

.PHONY: clean all 