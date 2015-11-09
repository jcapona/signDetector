CXX=g++ -std=c++11
# Includes, cflags
INCLUDES=-I../include -L../lib `pkg-config opencv --cflags` 
CXXOPTS=-Wall -g -O2
CXXFLAGS=$(CXXOPTS) $(INCLUDES)
# Libs flags
LDLIBS=-ltesseract `pkg-config opencv --libs`
# Binaries
TARGETS=signDetector

all:$(TARGETS)

clean:
	rm -f $(TARGETS)

default:
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

