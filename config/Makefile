# Auto-generated Makefile for independent chiplet performance testing
# Project environment
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/ai_sim_gen

# Compiler environment of C/C++
CC=g++
CFLAGS=-Wall -O3 -std=c++11 -fopenmp
LDFLAGS=-lm

# C/C++ Source files
CPU_SRCS=../simulators/devices/aerospace_cpu_task.cpp
CPU_OBJS=obj/aerospace_cpu_task.o
CPU_TARGET=bin/aerospace_cpu_task

# Additional device simulators
SENSOR_SRCS=../simulators/devices/sensor_simulator.cpp
COMM_SRCS=../simulators/devices/communication_simulator.cpp
CTRL_SRCS=../simulators/devices/controller_simulator.cpp

SENSOR_OBJS=obj/sensor_simulator.o
COMM_OBJS=obj/communication_simulator.o
CTRL_OBJS=obj/controller_simulator.o

SENSOR_TARGET=bin/sensor_simulator
COMM_TARGET=bin/communication_simulator
CTRL_TARGET=bin/controller_simulator

# Compiler environment of CUDA
NVCC=nvcc
CUFLAGS=--compiler-options -Wall -O3 -arch=sm_50

# CUDA Source files
GPU_SRCS=../simulators/devices/aerospace_gpu_task.cu
GPU_OBJS=cuobj/aerospace_gpu_task.o
GPU_TARGET=bin/aerospace_gpu_task

all: bin_dir obj_dir cuobj_dir CPU_target GPU_target device_targets

# C++ target for CPU tasks
CPU_target: $(CPU_OBJS)
	$(CC) $(CPU_OBJS) $(LDFLAGS) -o $(CPU_TARGET)

# CUDA target for GPU tasks
GPU_target: $(GPU_OBJS)
	$(NVCC) $(GPU_OBJS) -o $(GPU_TARGET)

# Device simulator targets
device_targets: $(SENSOR_OBJS) $(COMM_OBJS) $(CTRL_OBJS)
	$(CC) $(SENSOR_OBJS) $(LDFLAGS) -o $(SENSOR_TARGET)
	$(CC) $(COMM_OBJS) $(LDFLAGS) -o $(COMM_TARGET)
	$(CC) $(CTRL_OBJS) $(LDFLAGS) -o $(CTRL_TARGET)

# Rule for C++ object files
obj/aerospace_cpu_task.o: $(CPU_SRCS)
	$(CC) $(CFLAGS) -c $< -o $@

obj/sensor_simulator.o: $(SENSOR_SRCS)
	$(CC) $(CFLAGS) -c $< -o $@

obj/communication_simulator.o: $(COMM_SRCS)
	$(CC) $(CFLAGS) -c $< -o $@

obj/controller_simulator.o: $(CTRL_SRCS)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for CUDA object
cuobj/aerospace_gpu_task.o: $(GPU_SRCS)
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
