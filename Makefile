cur_dir = $(shell pwd)
CXXFLAGS += -Wall -Wextra -pedantic -Wshadow -O3 -std=c++11 -fopenmp -g -fPIC -lhwloc -I$(cur_dir)/src -I$(cur_dir)/test
LIB = -L$(cur_dir) -lallreduce -Wl,-rpath=$(cur_dir) -lrt

# NCCL2 is available at:
# NOTE: The current NCCL 2 we have is based on cuda 8.0
# - ray: /usr/workspace/wsb/brain/nccl2/nccl_2.0.5-3+cuda8.0_ppc64el
# - surface: /usr/workspace/wsb/brain/nccl2/nccl-2.0.5+cuda8.0
ifeq ($(shell hostname|grep ray -c), 1)
	ENABLE_CUDA = YES
	loadcuda = $(shell module load cuda/8.0)
	NCCL_DIR = /usr/workspace/wsb/brain/nccl2/nccl_2.0.5-3+cuda8.0_ppc64el
	CXXFLAGS += -I$(NCCL_DIR)/include  -DALUMINUM_HAS_NCCL
	LIB += -L$(NCCL_DIR)/lib -lnccl -Wl,-rpath=$(NCCL_DIR)/lib
endif
ifeq ($(shell hostname|grep surface -c), 1)
	ENABLE_CUDA = YES
	loadcuda = $(shell module load cuda/8.0)
	NCCL_DIR = /usr/workspace/wsb/brain/nccl2/nccl-2.0.5+cuda8.0
	CXXFLAGS += -I$(NCCL_DIR)/include  -DALUMINUM_HAS_NCCL
	LIB += -L$(NCCL_DIR)/lib -lnccl -Wl,-rpath=$(NCCL_DIR)/lib
endif

ifeq ($(ENABLE_CUDA), YES)
	CUDA_HOME = $(patsubst %/,%,$(dir $(patsubst %/,%,$(dir $(shell which nvcc)))))
	CXXFLAGS += -I$(CUDA_HOME)/include -DALUMINUM_HAS_CUDA 
	LIB += -L$(CUDA_HOME)/lib64 -lcudart -Wl,-rpath=$(CUDA_HOME)/lib64
endif

all: liballreduce.so benchmark_allreduces benchmark_nballreduces benchmark_overlap benchmark_reductions test_correctness test_multi_nballreduces

liballreduce.so: src/allreduce.cpp src/allreduce_mpi_impl.cpp src/allreduce.hpp src/allreduce_impl.hpp src/allreduce_mempool.hpp src/allreduce_mpi_impl.hpp src/tuning_params.hpp src/allreduce_nccl_impl.hpp src/allreduce_nccl_impl.cpp
	mpicxx $(CXXFLAGS) -shared -o liballreduce.so src/allreduce.cpp src/allreduce_mpi_impl.cpp src/allreduce_nccl_impl.cpp

benchmark_allreduces: liballreduce.so benchmark/benchmark_allreduces.cpp src/allreduce_nccl_impl.hpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_allreduces benchmark/benchmark_allreduces.cpp

benchmark_nballreduces: liballreduce.so benchmark/benchmark_nballreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_nballreduces benchmark/benchmark_nballreduces.cpp

benchmark_overlap: liballreduce.so benchmark/benchmark_overlap.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_overlap benchmark/benchmark_overlap.cpp

test_correctness: liballreduce.so test/test_correctness.cpp src/allreduce_nccl_impl.hpp
	mpicxx $(CXXFLAGS) $(LIB) -o test_correctness test/test_correctness.cpp

test_multi_nballreduces: liballreduce.so test/test_multi_nballreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o test_multi_nballreduces test/test_multi_nballreduces.cpp

benchmark_reductions: benchmark/benchmark_reductions.cpp
	mpicxx $(CXXFLAGS) -o benchmark_reductions benchmark/benchmark_reductions.cpp

clean:
	rm -f liballreduce.so benchmark_allreduces benchmark_nballreduces benchmark_reductions test_correctness test_multi_nballreduces benchmark_overlap
