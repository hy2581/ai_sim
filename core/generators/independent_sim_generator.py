#!/usr/bin/env python3
"""
Independent Chiplet Simulation Generator with Aerospace Enhancements
Generates simulation files for aerospace microsystem performance testing
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from jinja2 import Template
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.parsers.independent_chiplet_parser import SystemConfig


class AerospaceSimulationGenerator:
    def __init__(self, simulator_root: str):
        self.simulator_root = Path(simulator_root)
        
    def generate_yaml_config(self, config: SystemConfig, output_dir: Path) -> str:
        """Generate YAML configuration file for aerospace microsystem testing"""
        yaml_template = """# Auto-generated aerospace microsystem simulation configuration
# System: {{ config.cpu.cores }}-core {{ config.cpu.arch }} CPU, {{ config.gpu.sm_count }}SM {{ config.gpu.arch }} GPU
# Memory: {{ config.memory.size_gb }}GB {{ config.memory.type }}
# Aerospace Components: {{ config.sensors|length }} sensors, {{ config.communications|length }} comm, {{ config.controllers|length }} controllers
# Test Duration: {{ config.test_duration_sec }} seconds
# Environment: {{ config.environment_config.radiation_level }} radiation, {{ config.environment_config.temperature_c }}°C

# Phase 1 configuration - Core computing components
phase1:
  # CPU Task with Aerospace Reliability Features
  - cmd: "$BENCHMARK_ROOT/bin/aerospace_cpu_task"
    args: ["0", "{{ config.cpu.task.iterations }}", "{{ config.cpu.task.data_size }}", 
           "{{ config.cpu.aerospace_reliability.radiation_hardening|lower }}", 
           "{{ config.cpu.aerospace_reliability.seu_tolerance }}"]
    log: "aerospace_cpu_task.log"
    is_to_stdout: true
    clock_rate: 1
    environment:
      radiation_level: "{{ config.environment_config.radiation_level }}"
      temperature_c: {{ config.environment_config.temperature_c }}
      
  # GPU Tasks with Aerospace Reliability Features
  {% for i in range(3) %}
  - cmd: "$BENCHMARK_ROOT/bin/aerospace_gpu_task"
    args: ["{{ i+1 }}", "{{ config.gpu.task.iterations }}", "{{ config.gpu.task.data_size }}",
           "{{ config.gpu.aerospace_reliability.ecc_enabled|lower }}",
           "{{ config.gpu.aerospace_reliability.redundancy_level }}"]
    log: "aerospace_gpu_task_{{ i+1 }}.log"
    is_to_stdout: true
    clock_rate: 1
    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"
    environment:
      radiation_level: "{{ config.environment_config.radiation_level }}"
      temperature_c: {{ config.environment_config.temperature_c }}
  {% endfor %}

# Phase 2 configuration - Aerospace-specific components
phase2:
  # Sensor Simulation Tasks
  {% for sensor in config.sensors %}
  - cmd: "$BENCHMARK_ROOT/bin/sensor_simulator"
    args: ["{{ sensor.sensor_type }}", "{{ sensor.sampling_rate_hz }}", 
           "{{ sensor.resolution_bits }}", "{{ sensor.accuracy_percent }}"]
    log: "sensor_{{ sensor.sensor_type }}.log"
    is_to_stdout: true
    clock_rate: 1
    power_profile:
      max_power_mw: {{ sensor.power_consumption_mw }}
      idle_power_mw: {{ sensor.power_consumption_mw * 0.1 }}
  {% endfor %}
  
  # Communication System Tasks
  {% for comm in config.communications %}
  - cmd: "$BENCHMARK_ROOT/bin/communication_simulator"
    args: ["{{ comm.comm_type }}", "{{ comm.frequency_band }}", 
           "{{ comm.data_rate_mbps }}", "{{ comm.transmission_power_w }}"]
    log: "comm_{{ comm.comm_type }}.log"
    is_to_stdout: true
    clock_rate: 1
    power_profile:
      max_power_w: {{ comm.transmission_power_w }}
      idle_power_w: {{ comm.transmission_power_w * 0.05 }}
  {% endfor %}
  
  # Control System Tasks
  {% for controller in config.controllers %}
  - cmd: "$BENCHMARK_ROOT/bin/controller_simulator"
    args: ["{{ controller.controller_type }}", "{{ controller.control_frequency_hz }}", 
           "{{ controller.response_time_ms }}", "{{ controller.accuracy_degrees }}"]
    log: "controller_{{ controller.controller_type }}.log"
    is_to_stdout: true
    clock_rate: 1
    power_profile:
      max_power_w: {{ controller.power_consumption_w }}
      idle_power_w: {{ controller.power_consumption_w * 0.2 }}
  {% endfor %}

# Phase 3 configuration - Comprehensive analysis
phase3:
  # Aerospace Performance Analyzer with Power and AI Assessment
  - cmd: "python3"
    args: ["$BENCHMARK_ROOT/aerospace_performance_analyzer.py", "--comprehensive-analysis"]
    log: "aerospace_performance_analysis.log"
    is_to_stdout: true
    clock_rate: 1

# Configuration metadata
test_config:
  test_type: "aerospace_microsystem_simulation"
  cpu_task_type: "{{ config.cpu.task.task_type }}"
  gpu_task_type: "{{ config.gpu.task.task_type }}"
  test_duration_sec: {{ config.test_duration_sec }}
  performance_metrics: {{ config.performance_metrics }}
  aerospace_features:
    radiation_hardening: {{ config.cpu.aerospace_reliability.radiation_hardening|lower }}
    temperature_range: {{ config.cpu.aerospace_reliability.temperature_range }}
    redundancy_level: {{ config.cpu.aerospace_reliability.redundancy_level }}
    ecc_enabled: {{ config.memory.aerospace_reliability.ecc_enabled|lower }}
  environment:
    radiation_level: "{{ config.environment_config.radiation_level }}"
    temperature_c: {{ config.environment_config.temperature_c }}
    vibration_level: "{{ config.environment_config.vibration_level }}"
    vacuum_environment: {{ config.environment_config.vacuum_environment|lower }}
"""
        
        template = Template(yaml_template)
        yaml_content = template.render(config=config)
        
        # 保存到config目录
        config_dir = output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        yaml_file = config_dir / "aerospace_simulation.yml"
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
            
        return str(yaml_file)
    
    def generate_makefile(self, config: SystemConfig, output_dir: Path) -> str:
        """Generate enhanced Makefile for aerospace simulation"""
        makefile_content = """# Enhanced Makefile for Aerospace Microsystem Simulation
# Project environment
BENCHMARK_ROOT=$(SIMULATOR_ROOT)/benchmark/ai_sim_gen

# Compiler environment
CC=g++
NVCC=nvcc
CFLAGS=-Wall -O3 -std=c++11 -fopenmp -DAEROSPACE_FEATURES
NVCCFLAGS=-O3 -std=c++11 -DAEROSPACE_FEATURES
LDFLAGS=-lm -lgomp
CUDA_LDFLAGS=-lm -lcuda -lcudart

# Source files
CPU_SRCS=independent_cpu_task.cpp aerospace_cpu_task.cpp
GPU_SRCS=independent_gpu_task.cu aerospace_gpu_task.cu
SENSOR_SRCS=sensor_simulator.cpp
COMM_SRCS=communication_simulator.cpp
CONTROLLER_SRCS=controller_simulator.cpp

# Directories
OBJ_DIR=obj
BIN_DIR=bin
LOG_DIR=logs

.PHONY: all clean setup

all: setup aerospace_cpu_task sensor_simulator

setup:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) $(LOG_DIR)
	@echo "Setting up aerospace simulation environment..."

# CPU compilation rules
%.o: %.cpp
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -c $< -o $(OBJ_DIR)/$@

aerospace_cpu_task: aerospace_cpu_task.o
	@echo "Linking aerospace CPU task: $@"
	$(CC) $(OBJ_DIR)/$< $(LDFLAGS) -o $(BIN_DIR)/$@

sensor_simulator: sensor_simulator.o
	@echo "Linking sensor simulator: $@"
	$(CC) $(OBJ_DIR)/$< $(LDFLAGS) -o $(BIN_DIR)/$@

# GPU compilation rules (if CUDA is available)
%.cu.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $(OBJ_DIR)/$@

aerospace_gpu_task: aerospace_gpu_task.cu.o
	@echo "Linking aerospace GPU task: $@"
	$(NVCC) $(OBJ_DIR)/$< $(CUDA_LDFLAGS) -o $(BIN_DIR)/$@

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR)/* $(BIN_DIR)/* $(LOG_DIR)/*

test_aerospace:
	@echo "Running aerospace simulation test..."
	python3 aerospace_sim_runner.py
"""
        
        # 保存到config目录
        config_dir = output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        makefile_file = config_dir / "Makefile_aerospace"
        with open(makefile_file, 'w', encoding='utf-8') as f:
            f.write(makefile_content)
            
        return str(makefile_file)
    
    def generate_cpp_files(self, config: SystemConfig, output_dir: Path) -> Dict[str, str]:
        """Generate C++ source files for aerospace simulation"""
        generated_files = {}
        
        # 检查是否已存在aerospace_cpu_task.cpp
        cpu_file = output_dir / "aerospace_cpu_task.cpp"
        if cpu_file.exists():
            generated_files["aerospace_cpu_task"] = str(cpu_file)
        
        # 检查是否已存在sensor_simulator.cpp
        sensor_file = output_dir / "sensor_simulator.cpp"
        if sensor_file.exists():
            generated_files["sensor_simulator"] = str(sensor_file)
        
        # 如果需要，可以生成通信和控制器仿真器的占位符
        comm_content = """// Communication Simulator Placeholder
#include <iostream>
int main(int argc, char* argv[]) {
    std::cout << "Communication simulator placeholder" << std::endl;
    return 0;
}"""
        
        controller_content = """// Controller Simulator Placeholder  
#include <iostream>
int main(int argc, char* argv[]) {
    std::cout << "Controller simulator placeholder" << std::endl;
    return 0;
}"""
        
        # 生成通信仿真器
        comm_file = output_dir / "communication_simulator.cpp"
        if not comm_file.exists():
            with open(comm_file, 'w') as f:
                f.write(comm_content)
        generated_files["communication_simulator"] = str(comm_file)
        
        # 生成控制器仿真器
        controller_file = output_dir / "controller_simulator.cpp"
        if not controller_file.exists():
            with open(controller_file, 'w') as f:
                f.write(controller_content)
        generated_files["controller_simulator"] = str(controller_file)
        
        return generated_files


def main():
    """主函数 - 用于测试"""
    from independent_chiplet_parser import IndependentChipletParser
    
    # 示例配置
    api_key = "sk-c5446da718c24bfab2536bf0b2d5abb0"
    parser = IndependentChipletParser(api_key)
    
    description = "测试一个8核x86 CPU和80SM Ampere GPU的航空航天微系统，包含IMU传感器、射频通信和姿态控制器"
    
    config = parser.parse(description)
    if config:
        generator = AerospaceSimulationGenerator("/home/hao123/chiplet")
        output_dir = Path("/home/hao123/chiplet/benchmark/ai_sim_gen")
        
        yaml_file = generator.generate_yaml_config(config, output_dir)
        makefile = generator.generate_makefile(config, output_dir)
        cpp_files = generator.generate_cpp_files(config, output_dir)
        
        print("Generated aerospace simulation files:")
        print(f"  YAML: {yaml_file}")
        print(f"  Makefile: {makefile}")
        print(f"  C++ files: {cpp_files}")
    else:
        print("Failed to parse configuration")


if __name__ == "__main__":
    main()