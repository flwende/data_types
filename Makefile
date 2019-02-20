CXX = g++
LD = g++

INC = -I./include -I./src/include
CXXFLAGS = -O3 -std=c++17 -mavx2 -mfma -fopenmp -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized $(INC)
LDFLAGS = -O3 -fopenmp

#CXXFLAGS += -DCHECK_RESULTS
#CXXFLAGS += -DAOS_LAYOUT
CXXFLAGS += -DSOA_LAYOUT
#CXXFLAGS += -DELEMENT_ACCESS
CXXFLAGS += -DINPLACE

target: bin/test_buffer.x

bin/test_buffer.x: obj/test_buffer.o obj/kernel.o
	$(LD) $(LDFLAGS) -o $@ $^

obj/test_buffer.o: src/test_buffer.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

obj/kernel.o: src/kernel.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
	$(CXX) $(CXXFLAGS) -S $<

clean:
	rm -f *~ obj/*.o bin/test_buffer.x

