CXX = g++
LD = g++

#INTELROOT = $(MKLROOT)/../compiler
#INC = -I$(HOME)/opt/gnu-7.2.0/boost/include -I$(GLIBC_DIR)/include -I./include -I./src/include -I$(INTELROOT)/include
#CXXFLAGS = -O3 -std=c++14 -mavx2 -fopenmp -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized $(INC)
#LDFLAGS = -O2 -fopenmp -Wl,-rpath,$(GLIBC_DIR)/lib -Wl,-dynamic-linker,$(GLIBC_DIR)/lib/ld-linux-x86-64.so.2 -L$(GLIBC_DIR)/lib
INC = -I$(HOME)/opt/gnu-7.2.0/boost/include -I./include -I./src/include
CXXFLAGS = -O3 -std=c++14 -mavx2 -fopenmp -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized $(INC)
LDFLAGS = -O2 -fopenmp

CXXFLAGS += -DCHECK_RESULTS
CXXFLAGS += -DAOS_LAYOUT
#CXXFLAGS += -DSOA_LAYOUT
#CXXFLAGS += -DELEMENT_ACCESS

target: bin/test_buffer.x

bin/test_buffer.x: obj/test_buffer.o obj/kernel.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/test_buffer.o: src/test_buffer.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

obj/kernel.o: src/kernel.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	$(CXX) $(CXXFLAGS) -S $<

test_include: obj/test_include.o

obj/test_include.o: src/test_include.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	rm -f *~ obj/*.o bin/test_buffer.x

