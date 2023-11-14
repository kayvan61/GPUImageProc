CXX=nvcc

SRC_DIR := ./src
OBJ_DIR := ./objs
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
LDFLAGS := 
CPPFLAGS := 
CXXFLAGS := -g

all: conv histo

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