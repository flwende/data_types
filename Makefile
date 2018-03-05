CXX = g++
LD = g++

INC = -I$(HOME)/opt/gnu-7.2.0/boost/include -I$(GLIBC_DIR)/include -I./include
CXXFLAGS = -O3 -std=c++14 -mavx2 -fopenmp -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized $(INC)
LDFLAGS = -O2 -fopenmp -Wl,-rpath,$(GLIBC_DIR)/lib -Wl,-dynamic-linker,$(GLIBC_DIR)/lib/ld-linux-x86-64.so.2 -L$(GLIBC_DIR)/lib

target: bin/test_buffer.x

bin/test_buffer.x: obj/test_buffer.o
	$(LD) $(LDFLAGS) -o $@ $<

obj/test_buffer.o: src/test_buffer.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f *~ obj/*.o bin/test_buffer.x

