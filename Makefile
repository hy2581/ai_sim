# Auto-generated Makefile for independent chiplet performance testing
# Project environment
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/ai_sim_gen

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -O3 -std=c++11 -fopenmp
LDFLAGS=-lm

# C/C++ Source files
CPU_SRCS=independent_cpu_task.cpp
CPU_OBJS=obj/independent_cpu_task.o
CPU_TARGET=bin/independent_cpu_task

# Compiler environment of CUDA
NVCC=nvcc
CUFLAGS=--compiler-options -Wall -O3 -arch=sm_50

# CUDA Source files
GPU_SRCS=independent_gpu_task.cu
GPU_OBJS=cuobj/independent_gpu_task.o
GPU_TARGET=bin/independent_gpu_task

all: bin_dir obj_dir cuobj_dir CPU_target GPU_target

# C++ target for CPU tasks
CPU_target: $(CPU_OBJS)
	$(CC) $(CPU_OBJS) $(LDFLAGS) -o $(CPU_TARGET)

# CUDA target for GPU tasks
GPU_target: $(GPU_OBJS)
	$(NVCC) $(GPU_OBJS) -o $(GPU_TARGET)

# Rule for C++ object
obj/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for CUDA object
cuobj/%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

# Directory for binary files
bin_dir:
	mkdir -p bin

# Directory for object files for C++
obj_dir:
	mkdir -p obj

# Directory for object files for CUDA
cuobj_dir:
	mkdir -p cuobj

# Clean generated files
clean:
	rm -rf *_task_*_results.json *.log
	rm -rf proc_r*_t* performance_analysis_*.json
	rm -rf obj cuobj bin

# Test individual components
test_cpu:
	./bin/independent_cpu_task 0 10000 1024

test_gpu:
	./bin/independent_gpu_task 1 1000 4096

.PHONY: all CPU_target GPU_target clean bin_dir obj_dir cuobj_dir test_cpu test_gpu
