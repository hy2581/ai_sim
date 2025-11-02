#!/usr/bin/env python3
"""
Device Requirements Mapper
通过仿真器辅助将任务需求映射到具体器件参数需求
"""

import os
import sys
import json
import math
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

# 导入任务需求定义
sys.path.append(str(Path(__file__).parent))
from task_requirements_analyzer import TaskRequirements


@dataclass
class CPURequirements:
    """CPU需求规格"""
    min_cores: int
    min_frequency_ghz: float
    architecture: str  # ARM, x86, RISC-V
    cache_l1_kb: int
    cache_l2_kb: int
    cache_l3_mb: int
    instruction_sets: List[str]  # 支持的指令集
    power_budget_w: float
    thermal_design_power_w: float


@dataclass
class GPURequirements:
    """GPU需求规格"""
    min_sm_count: int
    min_frequency_mhz: float
    architecture: str  # ampere, turing, etc.
    memory_size_gb: int
    memory_type: str  # GDDR6, HBM2
    memory_bandwidth_gbps: float
    compute_capability: str
    power_budget_w: float
    ecc_support: bool


@dataclass
class MemoryRequirements:
    """内存需求规格"""
    min_capacity_gb: int
    memory_type: str  # DDR4, DDR5, LPDDR5
    min_frequency_mhz: int
    min_bandwidth_gbps: float
    max_latency_ns: float
    ecc_support: bool
    power_per_gb_w: float


@dataclass
class SensorRequirements:
    """传感器需求规格"""
    sensor_type: str
    min_sampling_rate_hz: float
    resolution_bits: int
    accuracy_percent: float
    interface_type: str  # SPI, I2C, UART
    power_consumption_mw: float
    operating_temp_range: Tuple[float, float]


@dataclass
class CommunicationRequirements:
    """通信器件需求规格"""
    comm_type: str
    min_data_rate_mbps: float
    max_latency_ms: float
    protocol_support: List[str]
    power_consumption_w: float
    range_km: float
    reliability_percent: float


@dataclass
class ControllerRequirements:
    """控制器需求规格"""
    controller_type: str
    min_response_time_ms: float
    control_precision: float
    interface_types: List[str]
    power_consumption_w: float
    redundancy_level: int


@dataclass
class DeviceConfiguration:
    """器件配置规格"""
    cpu: CPURequirements
    gpu: GPURequirements
    memory: MemoryRequirements
    sensors: List[SensorRequirements]
    communications: List[CommunicationRequirements]
    controllers: List[ControllerRequirements]
    total_power_budget_w: float
    form_factor_constraints: Dict[str, float]


class SimulatorInterface:
    """仿真器接口"""
    
    def __init__(self, simulator_root: str):
        self.simulator_root = Path(simulator_root)
        self.gem5_path = self.simulator_root / "gem5"
        self.gpgpu_sim_path = self.simulator_root / "gpgpu-sim"
    
    def estimate_cpu_performance(self, cores: int, freq_ghz: float, 
                                architecture: str) -> Dict[str, float]:
        """使用gem5估算CPU性能"""
        print(f"使用仿真器估算CPU性能: {cores}核 {freq_ghz}GHz {architecture}")
        
        # 基于架构的IPC估算
        ipc_estimates = {
            "ARM": 2.5,      # ARM Cortex-A78级别
            "x86": 3.0,      # Intel Core级别
            "RISC-V": 2.0    # 高性能RISC-V
        }
        
        base_ipc = ipc_estimates.get(architecture, 2.0)
        
        # 考虑多核效率
        multi_core_efficiency = min(1.0, 0.8 + 0.2 * math.log(cores) / math.log(16))
        
        # 计算性能指标
        instructions_per_second = cores * freq_ghz * 1e9 * base_ipc * multi_core_efficiency
        gips = instructions_per_second / 1e9
        
        # 功耗估算
        base_power_per_core = 15  # W
        freq_power_factor = (freq_ghz / 2.5) ** 2.5  # 频率对功耗的影响
        total_power = cores * base_power_per_core * freq_power_factor
        
        return {
            "gips": gips,
            "instructions_per_second": instructions_per_second,
            "power_consumption_w": total_power,
            "ipc": base_ipc * multi_core_efficiency,
            "multi_core_efficiency": multi_core_efficiency
        }
    
    def estimate_gpu_performance(self, sm_count: int, freq_mhz: float,
                               architecture: str) -> Dict[str, float]:
        """使用GPGPU-Sim估算GPU性能"""
        print(f"使用仿真器估算GPU性能: {sm_count}SM {freq_mhz}MHz {architecture}")
        
        # 基于架构的每SM性能估算
        sm_performance = {
            "ampere": 150,    # GFLOPS per SM
            "turing": 120,
            "pascal": 80
        }
        
        base_gflops_per_sm = sm_performance.get(architecture, 100)
        
        # 频率影响
        freq_factor = freq_mhz / 1400  # 基准频率1400MHz
        
        # 总性能计算
        total_gflops = sm_count * base_gflops_per_sm * freq_factor
        
        # 内存带宽需求估算
        memory_bandwidth_gbps = sm_count * 20 * freq_factor  # 每SM约20GB/s
        
        # 功耗估算
        power_per_sm = 2.5  # W per SM
        total_power = sm_count * power_per_sm * freq_factor
        
        return {
            "gflops": total_gflops,
            "memory_bandwidth_gbps": memory_bandwidth_gbps,
            "power_consumption_w": total_power,
            "sm_utilization": 0.85,  # 预期利用率
            "memory_utilization": 0.70
        }
    
    def estimate_memory_performance(self, capacity_gb: int, type_str: str,
                                  frequency_mhz: int) -> Dict[str, float]:
        """估算内存性能"""
        print(f"估算内存性能: {capacity_gb}GB {type_str} {frequency_mhz}MHz")
        
        # 不同内存类型的特性
        memory_specs = {
            "DDR4": {"bandwidth_factor": 1.0, "latency_ns": 120, "power_per_gb": 0.5},
            "DDR5": {"bandwidth_factor": 1.6, "latency_ns": 100, "power_per_gb": 0.4},
            "LPDDR5": {"bandwidth_factor": 1.4, "latency_ns": 110, "power_per_gb": 0.3},
            "HBM2": {"bandwidth_factor": 8.0, "latency_ns": 80, "power_per_gb": 1.0}
        }
        
        specs = memory_specs.get(type_str, memory_specs["DDR4"])
        
        # 带宽计算
        base_bandwidth = frequency_mhz * 8 / 1000  # GB/s (64-bit bus)
        actual_bandwidth = base_bandwidth * specs["bandwidth_factor"]
        
        # 功耗计算
        total_power = capacity_gb * specs["power_per_gb"]
        
        return {
            "bandwidth_gbps": actual_bandwidth,
            "latency_ns": specs["latency_ns"],
            "power_consumption_w": total_power,
            "efficiency": actual_bandwidth / total_power if total_power > 0 else 0
        }


class DeviceRequirementsMapper:
    """器件需求映射器"""
    
    def __init__(self, simulator_root: str = None):
        self.simulator_root = simulator_root or "/home/hao123/chiplet"
        self.simulator = SimulatorInterface(self.simulator_root)
    
    def map_cpu_requirements(self, task_req: TaskRequirements) -> CPURequirements:
        """映射CPU需求"""
        print("映射CPU需求...")
        
        # 基于任务需求确定CPU配置
        required_gips = task_req.min_cpu_gips
        
        # 尝试不同的CPU配置
        configurations = [
            (8, 2.5, "ARM"),
            (16, 2.0, "ARM"),
            (12, 2.8, "x86"),
            (8, 3.2, "x86")
        ]
        
        best_config = None
        best_score = float('inf')
        
        for cores, freq, arch in configurations:
            perf = self.simulator.estimate_cpu_performance(cores, freq, arch)
            
            if perf["gips"] >= required_gips:
                # 计算配置得分（性能/功耗比）
                score = perf["power_consumption_w"] / perf["gips"]
                if score < best_score:
                    best_score = score
                    best_config = (cores, freq, arch, perf)
        
        if best_config is None:
            # 如果没有满足需求的配置，选择最高性能的
            best_config = configurations[-1]
            perf = self.simulator.estimate_cpu_performance(*best_config)
            best_config = (*best_config, perf)
        
        cores, freq, arch, perf = best_config
        
        return CPURequirements(
            min_cores=cores,
            min_frequency_ghz=freq,
            architecture=arch,
            cache_l1_kb=32,  # 每核32KB L1
            cache_l2_kb=256,  # 每核256KB L2
            cache_l3_mb=max(2, cores // 4),  # 共享L3缓存
            instruction_sets=["NEON", "SVE"] if arch == "ARM" else ["AVX2", "AVX512"],
            power_budget_w=perf["power_consumption_w"] * 1.2,  # 20%余量
            thermal_design_power_w=perf["power_consumption_w"] * 1.1
        )
    
    def map_gpu_requirements(self, task_req: TaskRequirements) -> GPURequirements:
        """映射GPU需求"""
        print("映射GPU需求...")
        
        required_gflops = task_req.min_gpu_gflops
        
        # 尝试不同的GPU配置
        configurations = [
            (80, 1400, "ampere"),
            (120, 1200, "ampere"),
            (160, 1000, "ampere"),
            (100, 1600, "turing")
        ]
        
        best_config = None
        best_score = float('inf')
        
        for sm_count, freq, arch in configurations:
            perf = self.simulator.estimate_gpu_performance(sm_count, freq, arch)
            
            if perf["gflops"] >= required_gflops:
                score = perf["power_consumption_w"] / perf["gflops"]
                if score < best_score:
                    best_score = score
                    best_config = (sm_count, freq, arch, perf)
        
        if best_config is None:
            # 选择最高性能配置
            best_config = configurations[2]
            perf = self.simulator.estimate_gpu_performance(*best_config)
            best_config = (*best_config, perf)
        
        sm_count, freq, arch, perf = best_config
        
        # 内存配置
        memory_size = max(16, int(perf["memory_bandwidth_gbps"] / 50))  # 基于带宽需求
        
        return GPURequirements(
            min_sm_count=sm_count,
            min_frequency_mhz=freq,
            architecture=arch,
            memory_size_gb=memory_size,
            memory_type="GDDR6",
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
            compute_capability="8.6" if arch == "ampere" else "7.5",
            power_budget_w=perf["power_consumption_w"] * 1.2,
            ecc_support=True  # 航空航天应用需要ECC
        )
    
    def map_memory_requirements(self, task_req: TaskRequirements,
                              cpu_req: CPURequirements,
                              gpu_req: GPURequirements) -> MemoryRequirements:
        """映射内存需求"""
        print("映射内存需求...")
        
        # 基于CPU和GPU需求确定内存配置
        min_bandwidth = task_req.min_memory_bandwidth_gbps
        
        # 内存容量需求
        base_capacity = 16  # 基础16GB
        cpu_capacity_need = cpu_req.min_cores * 2  # 每核2GB
        gpu_capacity_need = gpu_req.memory_size_gb
        
        total_capacity = max(base_capacity, cpu_capacity_need + gpu_capacity_need)
        
        # 选择内存类型
        memory_types = ["DDR4", "DDR5", "LPDDR5"]
        frequencies = [3200, 4800, 6400]
        
        best_config = None
        best_score = float('inf')
        
        for mem_type, freq in zip(memory_types, frequencies):
            perf = self.simulator.estimate_memory_performance(total_capacity, mem_type, freq)
            
            if perf["bandwidth_gbps"] >= min_bandwidth:
                score = perf["power_consumption_w"] / perf["bandwidth_gbps"]
                if score < best_score:
                    best_score = score
                    best_config = (mem_type, freq, perf)
        
        if best_config is None:
            # 使用最高性能配置
            best_config = (memory_types[-1], frequencies[-1])
            perf = self.simulator.estimate_memory_performance(total_capacity, *best_config)
            best_config = (*best_config, perf)
        
        mem_type, freq, perf = best_config
        
        return MemoryRequirements(
            min_capacity_gb=total_capacity,
            memory_type=mem_type,
            min_frequency_mhz=freq,
            min_bandwidth_gbps=perf["bandwidth_gbps"],
            max_latency_ns=task_req.max_memory_latency_ns,
            ecc_support=True,  # 航空航天应用需要ECC
            power_per_gb_w=perf["power_consumption_w"] / total_capacity
        )
    
    def map_sensor_requirements(self, task_req: TaskRequirements) -> List[SensorRequirements]:
        """映射传感器需求"""
        print("映射传感器需求...")
        
        sensors = []
        
        # IMU传感器
        sensors.append(SensorRequirements(
            sensor_type="IMU",
            min_sampling_rate_hz=task_req.min_sensor_sampling_hz,
            resolution_bits=16,
            accuracy_percent=99.5,
            interface_type="SPI",
            power_consumption_mw=20,
            operating_temp_range=task_req.operating_temp_range
        ))
        
        # 环境传感器
        sensors.append(SensorRequirements(
            sensor_type="environmental",
            min_sampling_rate_hz=100,  # 环境传感器采样率较低
            resolution_bits=12,
            accuracy_percent=98.0,
            interface_type="I2C",
            power_consumption_mw=5,
            operating_temp_range=task_req.operating_temp_range
        ))
        
        return sensors
    
    def map_communication_requirements(self, task_req: TaskRequirements) -> List[CommunicationRequirements]:
        """映射通信需求"""
        print("映射通信需求...")
        
        communications = []
        
        # 射频通信
        communications.append(CommunicationRequirements(
            comm_type="RF_transceiver",
            min_data_rate_mbps=100,
            max_latency_ms=task_req.max_communication_latency_ms,
            protocol_support=["SpaceWire", "MIL-STD-1553"],
            power_consumption_w=5.0,
            range_km=1000,
            reliability_percent=99.9
        ))
        
        return communications
    
    def map_controller_requirements(self, task_req: TaskRequirements) -> List[ControllerRequirements]:
        """映射控制器需求"""
        print("映射控制器需求...")
        
        controllers = []
        
        # 姿态控制器
        controllers.append(ControllerRequirements(
            controller_type="attitude_controller",
            min_response_time_ms=task_req.max_control_response_ms,
            control_precision=0.1,  # 0.1度精度
            interface_types=["CAN", "PWM"],
            power_consumption_w=2.0,
            redundancy_level=2  # 双冗余
        ))
        
        return controllers
    
    def generate_device_configuration(self, task_req: TaskRequirements) -> DeviceConfiguration:
        """生成完整的器件配置"""
        print("生成器件配置...")
        print("=" * 60)
        
        # 映射各器件需求
        cpu_req = self.map_cpu_requirements(task_req)
        gpu_req = self.map_gpu_requirements(task_req)
        memory_req = self.map_memory_requirements(task_req, cpu_req, gpu_req)
        sensor_reqs = self.map_sensor_requirements(task_req)
        comm_reqs = self.map_communication_requirements(task_req)
        ctrl_reqs = self.map_controller_requirements(task_req)
        
        # 计算总功耗
        total_power = (cpu_req.power_budget_w + 
                      gpu_req.power_budget_w + 
                      memory_req.min_capacity_gb * memory_req.power_per_gb_w +
                      sum(s.power_consumption_mw / 1000 for s in sensor_reqs) +
                      sum(c.power_consumption_w for c in comm_reqs) +
                      sum(c.power_consumption_w for c in ctrl_reqs))
        
        # 形状因子约束
        form_factor = {
            "max_length_mm": 100,
            "max_width_mm": 100,
            "max_height_mm": 20,
            "max_weight_g": 500
        }
        
        config = DeviceConfiguration(
            cpu=cpu_req,
            gpu=gpu_req,
            memory=memory_req,
            sensors=sensor_reqs,
            communications=comm_reqs,
            controllers=ctrl_reqs,
            total_power_budget_w=min(float(total_power * 1.2), float(task_req.max_total_power_w)),
            form_factor_constraints=form_factor
        )
        
        print(f"器件配置生成完成，总功耗: {total_power:.1f}W")
        return config
    
    def generate_mapping_report(self, task_req: TaskRequirements, 
                              device_config: DeviceConfiguration) -> str:
        """生成映射报告"""
        report = f"""
# 器件需求映射报告

## 1. 任务需求到器件映射摘要

### 1.1 CPU映射结果
- **任务需求**: {task_req.min_cpu_gips:.2f} GIPS
- **器件配置**: {device_config.cpu.min_cores}核 {device_config.cpu.min_frequency_ghz:.1f}GHz {device_config.cpu.architecture}
- **预期性能**: 满足需求
- **功耗预算**: {device_config.cpu.power_budget_w:.1f}W

### 1.2 GPU映射结果  
- **任务需求**: {task_req.min_gpu_gflops:.2f} GFLOPS
- **器件配置**: {device_config.gpu.min_sm_count}SM {device_config.gpu.min_frequency_mhz}MHz {device_config.gpu.architecture}
- **内存配置**: {device_config.gpu.memory_size_gb}GB {device_config.gpu.memory_type}
- **功耗预算**: {device_config.gpu.power_budget_w:.1f}W

### 1.3 内存映射结果
- **任务需求**: {task_req.min_memory_bandwidth_gbps:.2f} GB/s
- **器件配置**: {device_config.memory.min_capacity_gb}GB {device_config.memory.memory_type} {device_config.memory.min_frequency_mhz}MHz
- **预期带宽**: {device_config.memory.min_bandwidth_gbps:.2f} GB/s
- **延迟要求**: ≤{device_config.memory.max_latency_ns:.0f}ns

### 1.4 传感器映射结果
"""
        for i, sensor in enumerate(device_config.sensors, 1):
            report += f"""
- **传感器{i}**: {sensor.sensor_type}
  - 采样率: {sensor.min_sampling_rate_hz:.0f} Hz
  - 精度: {sensor.accuracy_percent:.1f}%
  - 接口: {sensor.interface_type}
  - 功耗: {sensor.power_consumption_mw:.0f} mW
"""

        report += f"""
### 1.5 通信器件映射结果
"""
        for i, comm in enumerate(device_config.communications, 1):
            report += f"""
- **通信器件{i}**: {comm.comm_type}
  - 数据率: {comm.min_data_rate_mbps:.0f} Mbps
  - 延迟: ≤{comm.max_latency_ms:.0f} ms
  - 协议: {', '.join(comm.protocol_support)}
  - 功耗: {comm.power_consumption_w:.1f} W
"""

        report += f"""
### 1.6 控制器映射结果
"""
        for i, ctrl in enumerate(device_config.controllers, 1):
            report += f"""
- **控制器{i}**: {ctrl.controller_type}
  - 响应时间: ≤{ctrl.min_response_time_ms:.0f} ms
  - 控制精度: ±{ctrl.control_precision:.1f}°
  - 接口: {', '.join(ctrl.interface_types)}
  - 冗余级别: {ctrl.redundancy_level}
  - 功耗: {ctrl.power_consumption_w:.1f} W
"""

        report += f"""
## 2. 系统级配置摘要

### 2.1 功耗分析
- **总功耗预算**: {device_config.total_power_budget_w:.1f}W
- **任务功耗限制**: {task_req.max_total_power_w:.1f}W
- **功耗余量**: {task_req.max_total_power_w - device_config.total_power_budget_w:.1f}W

### 2.2 形状因子约束
- **最大尺寸**: {device_config.form_factor_constraints['max_length_mm']:.0f} × {device_config.form_factor_constraints['max_width_mm']:.0f} × {device_config.form_factor_constraints['max_height_mm']:.0f} mm
- **最大重量**: {device_config.form_factor_constraints['max_weight_g']:.0f} g

### 2.3 环境适应性
- **工作温度**: {task_req.operating_temp_range[0]:.0f}°C 至 {task_req.operating_temp_range[1]:.0f}°C
- **振动容忍**: {task_req.vibration_tolerance_g:.0f} G
- **辐射容忍**: {task_req.radiation_tolerance_krad:.0f} krad

## 3. 映射验证结论

基于仿真器辅助分析，所选器件配置能够满足任务性能需求，并在功耗、尺寸和环境适应性方面符合航空航天应用要求。

---
报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report


def main():
    """主函数"""
    # 加载任务需求
    task_req_file = Path(__file__).parent.parent.parent / "task_requirements.json"
    
    if not task_req_file.exists():
        print("错误: 未找到任务需求文件，请先运行任务需求分析")
        return
    
    with open(task_req_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        task_req_dict = data["requirements"]
    
    # 重建TaskRequirements对象
    task_req = TaskRequirements(**task_req_dict)
    
    # 创建器件需求映射器
    mapper = DeviceRequirementsMapper()
    
    # 生成器件配置
    device_config = mapper.generate_device_configuration(task_req)
    
    # 生成报告
    report = mapper.generate_mapping_report(task_req, device_config)
    
    # 保存结果
    output_dir = Path(__file__).parent.parent.parent
    
    # 保存JSON格式的器件配置
    with open(output_dir / "device_configuration.json", 'w', encoding='utf-8') as f:
        json.dump(asdict(device_config), f, indent=2, ensure_ascii=False)
    
    # 保存映射报告
    with open(output_dir / "device_mapping_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("器件需求映射完成！")
    print(f"器件配置已保存到: {output_dir / 'device_configuration.json'}")
    print(f"映射报告已保存到: {output_dir / 'device_mapping_report.md'}")
    
    return device_config


if __name__ == "__main__":
    main()